import OpenAI from "openai";
import { Pinecone } from "@pinecone-database/pinecone";

const SYSTEM_PROMPT =
  'You are a TED Talk assistant that answers questions strictly and only based on the TED dataset context provided to you (metadata and transcript passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: "I don’t know based on the provided TED data." Always explain your answer using the given context, quoting or paraphrasing the relevant transcript or metadata when helpful.';

function env(name) {
  const v = process.env[name];
  if (!v) throw new Error(`Missing env var: ${name}`);
  return v;
}

function formatContext(matches, maxChars = 1200) {
  const context = [];
  const parts = [];

  for (const m of matches || []) {
    const md = m?.metadata || {};
    const chunk = String(md.chunk || "").slice(0, maxChars);

    context.push({
      talk_id: md.talk_id,
      title: md.title,
      chunk,
      score: Number(m?.score || 0),
    });

    parts.push(
      `Talk ID: ${md.talk_id}\nTitle: ${md.title}\nChunk:\n${chunk}\n`
    );
  }

  return { ctxText: parts.join("\n---\n").trim(), ctxItems: context };
}

export default async function handler(req, res) {
  try {
    if (req.method !== "POST") {
      return res.status(405).json({ error: "Method not allowed" });
    }

    const body = typeof req.body === "string" ? JSON.parse(req.body) : req.body;
    const question = String(body?.question || "").trim();
    if (!question) {
      return res.status(400).json({ error: "Missing question" });
    }

    // Match your Python: TOP_K = 12
    const topK = parseInt(process.env.TOP_K || "12", 10);

    const openai = new OpenAI({
      apiKey: env("LLMSTUDIO_API_KEY"),
      baseURL: env("LLMSTUDIO_BASE_URL"),
    });

    const pc = new Pinecone({ apiKey: env("PINECONE_API_KEY") });

    // Namespace support (leave PINECONE_NAMESPACE unset if you used default namespace)
    const namespace = process.env.PINECONE_NAMESPACE;
    const index = namespace
      ? pc.index(env("PINECONE_INDEX_NAME")).namespace(namespace)
      : pc.index(env("PINECONE_INDEX_NAME"));

    // 1) Embed question
    const emb = await openai.embeddings.create({
      model: "RPRTHPB-text-embedding-3-small",
      input: question,
    });

    const qvec = emb?.data?.[0]?.embedding;
    if (!qvec) {
      return res.status(500).json({ error: "Failed to create embedding" });
    }

    // 2) Query Pinecone
    const queryRes = await index.query({
      vector: qvec,
      topK,
      includeMetadata: true,
    });

    const { ctxText, ctxItems } = formatContext(queryRes?.matches);

    // 3) Build augmented prompt
    const userPrompt =
      `Question:\n${question}\n\n` +
      `TED dataset context (top ${topK} retrieved chunks):\n${ctxText}\n\n` +
      "Answer using only the context above.";

    // 4) Generate answer (match your Python: temperature = 1)
    const completion = await openai.chat.completions.create({
      model: "RPRTHPB-gpt-5-mini",
      temperature: 1,
      messages: [
        { role: "system", content: SYSTEM_PROMPT },
        { role: "user", content: userPrompt },
      ],
    });

    const responseText = completion?.choices?.[0]?.message?.content || "";

    return res.status(200).json({
      response: responseText,
      context: ctxItems,
      Augmented_prompt: {
        System: SYSTEM_PROMPT,
        User: userPrompt,
      },
    });
  } catch (err) {
    return res.status(500).json({ error: err?.message || "Internal error" });
  }
}
