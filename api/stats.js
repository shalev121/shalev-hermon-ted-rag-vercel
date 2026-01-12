export default function handler(req, res) {
  if (req.method !== "GET") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const chunk_size = parseInt(process.env.CHUNK_SIZE || "1024", 10);
  const overlap_ratio = parseFloat(process.env.OVERLAP_RATIO || "0.15");
  const top_k = parseInt(process.env.TOP_K || "12", 10);


  return res.status(200).json({
    chunk_size,
    overlap_ratio,
    top_k
  });
}
