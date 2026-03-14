using Qdrant.Client;
using Qdrant.Client.Grpc;
using OllamaSharp;
using OllamaSharp.Models;
using DotNetEnv;
using UglyToad.PdfPig;
using System.Text;

class Program
{
    // ── UI Helpers ─────────────────────────────────────────────────
    static void Header()
    {
        Console.Clear();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║       🧠  RAG HYBRID SEARCH ENGINE  🧠                  ║");
        Console.WriteLine("║       PDF Chunking + Vector + BM25 Keyword Search       ║");
        Console.WriteLine("╚══════════════════════════════════════════════════════════╝");
        Console.ResetColor();
    }

    static void PrintSuccess(string msg)
    {
        Console.ForegroundColor = ConsoleColor.Green;
        Console.WriteLine($"  ✅ {msg}");
        Console.ResetColor();
    }

    static void PrintInfo(string msg)
    {
        Console.ForegroundColor = ConsoleColor.Yellow;
        Console.WriteLine($"  ⚡ {msg}");
        Console.ResetColor();
    }

    static void PrintSection(string title)
    {
        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Magenta;
        Console.WriteLine($"  ── {title} ──────────────────────────────");
        Console.ResetColor();
    }

    static void PrintResult(int rank, float score, string text, string source)
    {
        int barLength = (int)(score * 20);
        string bar = new string('█', barLength) + new string('░', 20 - barLength);

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.White;
        Console.Write($"  #{rank} ");

        if (score > 0.7f)      Console.ForegroundColor = ConsoleColor.Green;
        else if (score > 0.5f) Console.ForegroundColor = ConsoleColor.Yellow;
        else                   Console.ForegroundColor = ConsoleColor.Red;

        Console.Write($"[{bar}] {score:F3}");
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($" ({source})");
        Console.ResetColor();

        // Truncate long chunks for display
        // Lange Chunks für die Anzeige kürzen
        var display = text.Length > 200 ? text[..200] + "..." : text;
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine($"     {display}");
        Console.ResetColor();
    }

    // ── Chunking ───────────────────────────────────────────────────
    static List<string> ChunkText(string text, int chunkSize = 40, int overlap = 5)
    {
        // Fix missing spaces between words — common in PDF extraction
        // Fehlende Leerzeichen zwischen Wörtern reparieren — häufig bei PDF-Extraktion
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([a-z])([A-Z])", "$1 $2");
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"(\.)([A-Z])", "$1 $2");
        // Fix number+letter joins: "Volume1Long" → "Volume 1 Long"
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([0-9])([A-Za-z])", "$1 $2");
        // Fix letter+number joins: "pages1601" → "pages 1601"
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([A-Za-z])([0-9])", "$1 $2");
        // Fix bracket+letter joins: "(Volume" → "( Volume"
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([\(\[\{])([A-Za-z0-9])", "$1 $2");    

        var words = text.Split(' ', StringSplitOptions.RemoveEmptyEntries);
        var chunks = new List<string>();
        int i = 0;

        while (i < words.Length)
        {
            var chunkWords = words.Skip(i).Take(chunkSize).ToArray();
            if (chunkWords.Length > 0)
                chunks.Add(string.Join(" ", chunkWords));
            i += chunkSize - overlap;
        }

        return chunks;
    }

    // ── BM25 Keyword Search ────────────────────────────────────────
    static List<(int index, float score)> BM25Search(
        List<string> chunks, string query, int topK = 5)
    {
        // BM25 parameters — standard defaults used everywhere
        // BM25-Parameter — überall verwendete Standardwerte
        const float k1 = 1.5f;   // term frequency saturation
        const float b  = 0.75f;  // length normalization

        var queryTerms = query.ToLower()
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);

        // Average chunk length for length normalization
        // Durchschnittliche Chunk-Länge für Längennormalisierung
        double avgLen = chunks.Average(c =>
            c.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length);

        var scores = new List<(int index, float score)>();

        for (int i = 0; i < chunks.Count; i++)
        {
            var chunkTerms = chunks[i].ToLower()
                .Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int   chunkLen = chunkTerms.Length;
            float score    = 0;

            foreach (var term in queryTerms)
            {
                // Term frequency — how many times does term appear in chunk?
                // Termhäufigkeit — wie oft erscheint der Begriff im Chunk?
                int tf = chunkTerms.Count(t => t == term);
                if (tf == 0) continue;

                // Document frequency — how many chunks contain this term?
                // Dokumenthäufigkeit — wie viele Chunks enthalten diesen Begriff?
                int df = chunks.Count(c => c.ToLower().Contains(term));

                // IDF — rare terms score higher than common ones
                // IDF — seltene Begriffe erzielen höhere Punktzahl
                float idf = (float)Math.Log(
                    (chunks.Count - df + 0.5) / (df + 0.5) + 1);

                // TF with length normalization
                // TF mit Längennormalisierung
                float tfNorm = tf * (k1 + 1) /
                    (tf + k1 * (1f - b + b * (float)chunkLen / (float)avgLen));

                score += idf * tfNorm;
            }

            scores.Add((i, score));
        }

        return scores
            .Where(s => s.score > 0)
            .OrderByDescending(s => s.score)
            .Take(topK)
            .ToList();
    }

    // ── Main ───────────────────────────────────────────────────────
    static async Task Main(string[] args)
    {
        Header();

        // ── Connect ────────────────────────────────────────────────
        PrintSection("CONNECTING");

        Env.Load();
        var qdrantUrl = Environment.GetEnvironmentVariable("QDRANT_URL")!;
        var qdrantKey = Environment.GetEnvironmentVariable("QDRANT_API_KEY")!;

        var ollamaClient = new OllamaApiClient(new Uri("http://localhost:11434"));
        var uri          = new Uri(qdrantUrl);
        var qdrantClient = new QdrantClient(
            host: uri.Host, https: true, apiKey: qdrantKey);

        PrintSuccess("Ollama connected");
        PrintSuccess("Qdrant Cloud connected");

        // ── PDF Loading ────────────────────────────────────────────
        PrintSection("PDF LOADING");

        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.Write("  📂 Enter PDF path (or press Enter for default): ");
        Console.ResetColor();

        var pdfPath = Console.ReadLine()?.Trim();
        if (string.IsNullOrEmpty(pdfPath))
            pdfPath = "D:\\QdrantRAG\\document.pdf";

        if (!File.Exists(pdfPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"  ❌ File not found: {pdfPath}");
            Console.ResetColor();
            return;
        }

        PrintInfo("Reading PDF...");
        var fullText  = new StringBuilder();
        int pageCount = 0;

        // Extract text page by page
        // Text Seite für Seite extrahieren
        using (var pdf = PdfDocument.Open(pdfPath))
        {
            foreach (var page in pdf.GetPages())
            {
                fullText.Append(page.Text);
                fullText.Append(" ");
                pageCount++;
            }
        }

        PrintSuccess($"PDF loaded — {pageCount} pages, {fullText.Length:N0} characters");

        // ── Chunking ───────────────────────────────────────────────
        PrintSection("CHUNKING");
        PrintInfo("Splitting document into overlapping chunks...");

        // ✅ FIXED — was 150/20, now 40/5 to fit Ollama context limit
        var chunks = ChunkText(fullText.ToString(), chunkSize: 40, overlap: 5);

        PrintSuccess($"Created {chunks.Count} chunks (40 words each, 5 word overlap)");

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($"\n  Sample chunk #1:");
        Console.WriteLine($"  \"{chunks[0][..Math.Min(150, chunks[0].Length)]}...\"");
        Console.ResetColor();

        // ── Embedding ──────────────────────────────────────────────
        PrintSection("CREATING EMBEDDINGS");
        PrintInfo($"Embedding {chunks.Count} chunks via Ollama...");
        PrintInfo("This may take a few minutes for large PDFs.\n");

        // Delete old collection and create fresh one
        // Alte Collection löschen und neue erstellen
        const string collectionName = "pdf_documents";
        var collections = await qdrantClient.ListCollectionsAsync();
        if (collections.Any(c => c == collectionName))
            await qdrantClient.DeleteCollectionAsync(collectionName);

        await qdrantClient.CreateCollectionAsync(collectionName,
            new VectorParams { Size = 768, Distance = Distance.Cosine });

        var points = new List<PointStruct>();

        for (int i = 0; i < chunks.Count; i++)
        {
            // Animated progress bar / Animierter Fortschrittsbalken
            int    progress = (int)(((float)(i + 1) / chunks.Count) * 30);
            string bar      = new string('█', progress) + new string('░', 30 - progress);

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write($"\r  [{bar}] {i + 1}/{chunks.Count}  ");
            Console.ResetColor();

            var embeddingResponse = await ollamaClient.EmbedAsync(new EmbedRequest
            {
                Model = "nomic-embed-text",
                Input = new List<string> { chunks[i] }
            });

            var vector = embeddingResponse.Embeddings[0]
                .Select(x => (float)x).ToArray();

            points.Add(new PointStruct
            {
                Id      = new PointId { Num = (ulong)i },
                Vectors = vector,
                Payload =
                {
                    ["text"]        = chunks[i],
                    ["chunk_index"] = i
                }
            });
        }

        Console.WriteLine();
        await qdrantClient.UpsertAsync(collectionName, points);
        PrintSuccess($"All {chunks.Count} chunks stored in Qdrant!");

        // ── Hybrid Search Loop ─────────────────────────────────────
        PrintSection("HYBRID SEARCH READY");
        Console.ForegroundColor = ConsoleColor.Gray;
        Console.WriteLine("  Combining vector search + BM25 keyword search.");
        Console.WriteLine("  Type a question. Type 'EXIT' to quit.\n");
        Console.ResetColor();

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("  🔍 Your query: ");
            Console.ResetColor();

            var query = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(query)) continue;
            if (query.ToUpper() == "EXIT") break;

            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("  ⚡ Running hybrid search");
            for (int i = 0; i < 3; i++)
            {
                await Task.Delay(200);
                Console.Write(".");
            }
            Console.WriteLine();
            Console.ResetColor();

            // Vector search
            // Vektorsuche
            var queryEmbedding = await ollamaClient.EmbedAsync(new EmbedRequest
            {
                Model = "nomic-embed-text",
                Input = new List<string> { query }
            });

            var queryVector = queryEmbedding.Embeddings[0]
                .Select(x => (float)x).ToArray();

            var vectorResults = await qdrantClient.SearchAsync(
                collectionName, queryVector, limit: 5);

            // BM25 keyword search
            // BM25 Keyword-Suche
            var bm25Results = BM25Search(chunks, query, topK: 5);

            // Reciprocal Rank Fusion — combine both result lists
            // Reciprocal Rank Fusion — beide Ergebnislisten kombinieren
            var combinedScores = new Dictionary<int, float>();

            int vectorRank = 1;
            foreach (var r in vectorResults)
            {
                int idx = (int)r.Id.Num;
                combinedScores[idx] = combinedScores
                    .GetValueOrDefault(idx, 0) + 1f / (60 + vectorRank++);
            }

            int bm25Rank = 1;
            foreach (var (idx, _) in bm25Results)
            {
                combinedScores[idx] = combinedScores
                    .GetValueOrDefault(idx, 0) + 1f / (60 + bm25Rank++);
            }

            var finalResults = combinedScores
                .OrderByDescending(kv => kv.Value)
                .Take(3)
                .ToList();

            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"\n  📚 Top 3 hybrid results for: \"{query}\"");
            Console.ResetColor();

            int rank = 1;
            foreach (var (idx, score) in finalResults)
            {
                float normalized = Math.Min(score * 50, 1.0f);

                bool   inVector = vectorResults.Any(r => (int)r.Id.Num == idx);
                bool   inBm25   = bm25Results.Any(r => r.index == idx);
                string source   = (inVector && inBm25) ? "Vector + BM25 ⭐"
                                : inVector             ? "Vector only"
                                :                        "BM25 only";

                PrintResult(rank++, normalized, chunks[idx], source);
            }

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine("  ──────────────────────────────────────────────────");
            Console.ResetColor();
        }

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("  👋 Auf Wiedersehen! Day 2+3 complete.");
        Console.ResetColor();
        Console.WriteLine();
    }
}
