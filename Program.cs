using Qdrant.Client;
using Qdrant.Client.Grpc;
using OllamaSharp;
using OllamaSharp.Models;
using DotNetEnv;
using UglyToad.PdfPig;
using System.Text;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

class Program
{
    // ── UI Helpers ─────────────────────────────────────────────────
    static void Header()
    {
        Console.Clear();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("╔══════════════════════════════════════════════════════════╗");
        Console.WriteLine("║        🧠  FULL RAG PIPELINE  🧠                         ║");
        Console.WriteLine("║        PDF · Embed · BM25 · Rerank · Generate           ║");
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

    static void PrintResult(int rank, float score,
        string text, string source)
    {
        int    barLength = (int)(score * 20);
        string bar       = new string('█', barLength) +
                           new string('░', 20 - barLength);

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.White;
        Console.Write($"  #{rank} ");

        if (score > 0.7f)      Console.ForegroundColor = ConsoleColor.Green;
        else if (score > 0.5f) Console.ForegroundColor = ConsoleColor.Yellow;
        else                   Console.ForegroundColor = ConsoleColor.Red;

        Console.Write($"[{bar}] {score:F3} 🎯");
        Console.ResetColor();

        Console.ForegroundColor = ConsoleColor.DarkGray;
        Console.WriteLine($" ({source})");
        Console.ResetColor();

        var display = text.Length > 200 ? text[..200] + "..." : text;
        Console.ForegroundColor = ConsoleColor.White;
        Console.WriteLine($"     {display}");
        Console.ResetColor();
    }

    // ── Chunking ───────────────────────────────────────────────────
    static List<string> ChunkText(string text,
        int chunkSize = 40, int overlap = 5)
    {
        // Fix PDF ligatures — fi/fl stored as single special characters
        // PDF-Ligaturen reparieren — fi/fl als einzelne Sonderzeichen gespeichert
        text = text.Replace("ﬁ", "fi")
                   .Replace("ﬂ", "fl")
                   .Replace("ﬀ", "ff")
                   .Replace("ﬃ", "ffi")
                   .Replace("ﬄ", "ffl");

        // Remove URLs and citation numbers — reduce bibliography noise
        // URLs und Zitatnummern entfernen — Bibliographie-Rauschen reduzieren
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"https?://\S+", " ");
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"\[\d+\]", " ");

        // Fix missing spaces between words
        // Fehlende Leerzeichen zwischen Wörtern reparieren
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([a-z])([A-Z])", "$1 $2");
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"(\.)([A-Z])", "$1 $2");
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([0-9])([A-Za-z])", "$1 $2");
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([A-Za-z])([0-9])", "$1 $2");
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([\(\[\{])([A-Za-z0-9])", "$1 $2");
        text = System.Text.RegularExpressions.Regex.Replace(
            text, @"([a-z]{2})([A-Z][a-z])", "$1 $2");

        var words  = text.Split(' ',
            StringSplitOptions.RemoveEmptyEntries);
        var chunks = new List<string>();
        int i      = 0;

        while (i < words.Length)
        {
            var chunkWords = words.Skip(i).Take(chunkSize).ToArray();
            if (chunkWords.Length > 0)
                chunks.Add(string.Join(" ", chunkWords));
            i += chunkSize - overlap;
        }

        return chunks;
    }

    // ── BM25 ───────────────────────────────────────────────────────
    static List<(int index, float score)> BM25Search(
        List<string> chunks, string query, int topK = 5)
    {
        const float k1 = 1.5f;
        const float b  = 0.75f;

        var queryTerms = query.ToLower()
            .Split(' ', StringSplitOptions.RemoveEmptyEntries);

        double avgLen = chunks.Average(c =>
            c.Split(' ', StringSplitOptions.RemoveEmptyEntries).Length);

        var scores = new List<(int index, float score)>();

        for (int i = 0; i < chunks.Count; i++)
        {
            var   chunkTerms = chunks[i].ToLower()
                .Split(' ', StringSplitOptions.RemoveEmptyEntries);
            int   chunkLen   = chunkTerms.Length;
            float score      = 0;

            foreach (var term in queryTerms)
            {
                int tf = chunkTerms.Count(t => t == term);
                if (tf == 0) continue;

                int df = chunks.Count(c =>
                    c.ToLower().Contains(term));

                float idf = (float)Math.Log(
                    (chunks.Count - df + 0.5) / (df + 0.5) + 1);

                float tfNorm = tf * (k1 + 1) /
                    (tf + k1 * (1f - b + b *
                    (float)chunkLen / (float)avgLen));

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

    // ── Tokenizer ──────────────────────────────────────────────────
    static Dictionary<string, int> LoadVocab(string vocabPath)
    {
        var vocab = new Dictionary<string, int>();
        var lines = File.ReadAllLines(vocabPath);
        for (int i = 0; i < lines.Length; i++)
            vocab[lines[i]] = i;
        return vocab;
    }

    static List<long> TokenizeText(string text,
        Dictionary<string, int> vocab)
    {
        var tokenIds = new List<long>();
        foreach (var word in text.ToLower()
            .Split(' ', StringSplitOptions.RemoveEmptyEntries))
        {
            tokenIds.Add(vocab.TryGetValue(word, out int id)
                ? id : vocab["[UNK]"]);
        }
        return tokenIds;
    }

    // ── Cross-Encoder ──────────────────────────────────────────────
    static float RunCrossEncoder(InferenceSession session,
        long[] inputIds, long[] tokenTypeIds)
    {
        int seqLen        = inputIds.Length;
        var attentionMask = Enumerable.Repeat(1L, seqLen).ToArray();
        var shape         = new[] { 1, seqLen };

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input_ids",
                new DenseTensor<long>(inputIds, shape)),
            NamedOnnxValue.CreateFromTensor("attention_mask",
                new DenseTensor<long>(attentionMask, shape)),
            NamedOnnxValue.CreateFromTensor("token_type_ids",
                new DenseTensor<long>(tokenTypeIds, shape))
        };

        using var results = session.Run(inputs);
        float logit = results.First().AsEnumerable<float>().First();

        // Sigmoid: converts raw logit to 0-1 probability
        // Sigmoid: wandelt rohen Logit in 0-1 Wahrscheinlichkeit um
        return 1f / (1f + MathF.Exp(-logit));
    }

    // ── Rerank ─────────────────────────────────────────────────────
    static List<(int index, float score)> Rerank(
        InferenceSession session,
        List<string> candidateChunks,
        List<int> candidateIndices,
        string query,
        Dictionary<string, int> vocab)
    {
        var results = new List<(int index, float score)>();

        for (int i = 0; i < candidateChunks.Count; i++)
        {
            var inputIds     = new List<long>();
            var tokenTypeIds = new List<long>();

            // [CLS] query [SEP] chunk [SEP]
            inputIds.Add(vocab["[CLS]"]); tokenTypeIds.Add(0);

            foreach (var id in TokenizeText(query, vocab))
            { inputIds.Add(id); tokenTypeIds.Add(0); }

            inputIds.Add(vocab["[SEP]"]); tokenTypeIds.Add(0);

            foreach (var id in TokenizeText(candidateChunks[i], vocab))
            { inputIds.Add(id); tokenTypeIds.Add(1); }

            inputIds.Add(vocab["[SEP]"]); tokenTypeIds.Add(1);

            // Truncate to BERT max length
            // Auf BERT-Maximallänge kürzen
            const int maxLen = 512;
            if (inputIds.Count > maxLen)
            {
                inputIds     = inputIds.Take(maxLen).ToList();
                tokenTypeIds = tokenTypeIds.Take(maxLen).ToList();
            }

            float score = RunCrossEncoder(
                session,
                inputIds.ToArray(),
                tokenTypeIds.ToArray());

            results.Add((candidateIndices[i], score));

            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.Write(
                $"\r  ⚡ Reranking {i + 1}/{candidateChunks.Count}...");
            Console.ResetColor();
        }

        Console.WriteLine();
        return results.OrderByDescending(r => r.score).ToList();
    }

    // ── Answer Generation ──────────────────────────────────────────
    static async Task<string> GenerateAnswer(
        OllamaApiClient ollamaClient,
        string query,
        List<string> contextChunks)
    {
        // Build context block from top reranked chunks
        // Kontext-Block aus den Top-Reranked-Chunks erstellen
        var context = new StringBuilder();
        for (int i = 0; i < contextChunks.Count; i++)
        {
            context.AppendLine($"[Context {i + 1}]");
            context.AppendLine(contextChunks[i]);
            context.AppendLine();
        }

        // Prompt — grounded generation, no hallucination
        // Prompt — fundierte Generierung, keine Halluzination
        var prompt = $"""
            You are a helpful assistant. Answer the question using
            ONLY the context provided below. If the context does
            not contain enough information, say exactly:
            "I cannot find this in the document."

            Do not add any information beyond what is in the context.
            Keep your answer concise and clear.

            {context}
            Question: {query}

            Answer:
            """;

        var answer = new StringBuilder();

        // Stream tokens to console as Mistral generates them
        // Tokens zur Konsole streamen während Mistral sie generiert
        await foreach (var token in ollamaClient.GenerateAsync(
            new GenerateRequest
            {
                Model  = "mistral",
                Prompt = prompt,
                Stream = true
            }))
        {
            if (token?.Response != null)
            {
                answer.Append(token.Response);
                Console.ForegroundColor = ConsoleColor.White;
                Console.Write(token.Response);
                Console.ResetColor();
            }
        }

        Console.WriteLine();
        return answer.ToString();
    }

    // ── Main ───────────────────────────────────────────────────────
    static async Task Main(string[] args)
    {
        Header();

        // ── Load config ────────────────────────────────────────────
        Env.Load();
        var qdrantUrl = Environment.GetEnvironmentVariable("QDRANT_URL")!;
        var qdrantKey = Environment.GetEnvironmentVariable("QDRANT_API_KEY")!;
        var pdfPath   = Environment.GetEnvironmentVariable("PDF_PATH")
                        ?? "D:\\QdrantRAG\\document.pdf";

        // ── Connect ────────────────────────────────────────────────
        PrintSection("CONNECTING");

        var ollamaClient = new OllamaApiClient(
            new Uri("http://localhost:11434"));
        var uri          = new Uri(qdrantUrl);
        var qdrantClient = new QdrantClient(
            host: uri.Host, https: true, apiKey: qdrantKey);

        PrintSuccess("Ollama connected");
        PrintSuccess("Qdrant Cloud connected");

        // ── Load Reranker ──────────────────────────────────────────
        PrintSection("LOADING RERANKER");

        const string modelPath = "D:\\models\\reranker.onnx";
        const string vocabPath = "D:\\models\\vocab.txt";

        if (!File.Exists(modelPath) || !File.Exists(vocabPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine("  ❌ Reranker model not found at D:\\models\\");
            Console.WriteLine("  Run the download commands first.");
            Console.ResetColor();
            return;
        }

        var vocab   = LoadVocab(vocabPath);
        var session = new InferenceSession(modelPath);

        PrintSuccess($"Vocabulary loaded — {vocab.Count:N0} tokens");
        PrintSuccess("Cross-encoder model loaded (ms-marco-MiniLM-L-6-v2)");

        // ── PDF Loading ────────────────────────────────────────────
        PrintSection("PDF LOADING");

        if (!File.Exists(pdfPath))
        {
            Console.ForegroundColor = ConsoleColor.Red;
            Console.WriteLine($"  ❌ File not found: {pdfPath}");
            Console.WriteLine("  Set PDF_PATH in your .env file.");
            Console.ResetColor();
            return;
        }

        PrintInfo($"Loading: {Path.GetFileName(pdfPath)}");

        var fullText  = new StringBuilder();
        int pageCount = 0;

        using (var pdf = PdfDocument.Open(pdfPath))
        {
            foreach (var page in pdf.GetPages())
            {
                fullText.Append(page.Text);
                fullText.Append(" ");
                pageCount++;
            }
        }

        PrintSuccess($"Loaded {Path.GetFileName(pdfPath)} " +
            $"— {pageCount} pages, {fullText.Length:N0} characters");

        // ── Chunking ───────────────────────────────────────────────
        PrintSection("CHUNKING");
        PrintInfo("Splitting into overlapping chunks...");

        var chunks = ChunkText(fullText.ToString(),
            chunkSize: 40, overlap: 5);

        PrintSuccess($"Created {chunks.Count} chunks " +
            $"(40 words each, 5 word overlap)");

        // ── Embedding ──────────────────────────────────────────────
        PrintSection("CREATING EMBEDDINGS");
        PrintInfo($"Embedding {chunks.Count} chunks via Ollama...\n");

        const string collectionName = "pdf_documents";
        var collections = await qdrantClient.ListCollectionsAsync();
        if (collections.Any(c => c == collectionName))
            await qdrantClient.DeleteCollectionAsync(collectionName);

        await qdrantClient.CreateCollectionAsync(collectionName,
            new VectorParams { Size = 768, Distance = Distance.Cosine });

        var points = new List<PointStruct>();

        for (int i = 0; i < chunks.Count; i++)
        {
            int    progress = (int)(((float)(i + 1) / chunks.Count) * 30);
            string bar      = new string('█', progress) +
                              new string('░', 30 - progress);

            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write($"\r  [{bar}] {i + 1}/{chunks.Count}  ");
            Console.ResetColor();

            var embeddingResponse = await ollamaClient.EmbedAsync(
                new EmbedRequest
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

        // ── Search + Generate Loop ─────────────────────────────────
        PrintSection("RAG PIPELINE READY");
        Console.ForegroundColor = ConsoleColor.Gray;
        Console.WriteLine("  Ask any question about your document.");
        Console.WriteLine("  Type 'EXIT' to quit.\n");
        Console.ResetColor();

        while (true)
        {
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.Write("  🔍 Your question: ");
            Console.ResetColor();

            var query = Console.ReadLine()?.Trim();
            if (string.IsNullOrEmpty(query)) continue;
            if (query.ToUpper() == "EXIT") break;

            // ── Step 1: Vector search ──────────────────────────────
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.Write("\n  ⚡ Retrieving relevant chunks");
            for (int i = 0; i < 3; i++)
            {
                await Task.Delay(200);
                Console.Write(".");
            }
            Console.WriteLine();
            Console.ResetColor();

            var queryEmbedding = await ollamaClient.EmbedAsync(
                new EmbedRequest
                {
                    Model = "nomic-embed-text",
                    Input = new List<string> { query }
                });

            var queryVector = queryEmbedding.Embeddings[0]
                .Select(x => (float)x).ToArray();

            var vectorResults = await qdrantClient.SearchAsync(
                collectionName, queryVector, limit: 5);

            // ── Step 2: BM25 ───────────────────────────────────────
            var bm25Results = BM25Search(chunks, query, topK: 5);

            // ── Step 3: RRF → top 5 candidates ────────────────────
            var combinedScores = new Dictionary<int, float>();

            int vectorRank = 1;
            foreach (var r in vectorResults)
            {
                int idx = (int)r.Id.Num;
                combinedScores[idx] = combinedScores
                    .GetValueOrDefault(idx, 0) +
                    1f / (60 + vectorRank++);
            }

            int bm25Rank = 1;
            foreach (var (idx, _) in bm25Results)
            {
                combinedScores[idx] = combinedScores
                    .GetValueOrDefault(idx, 0) +
                    1f / (60 + bm25Rank++);
            }

            var top5 = combinedScores
                .OrderByDescending(kv => kv.Value)
                .Take(5)
                .ToList();

            // ── Step 4: Rerank ─────────────────────────────────────
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("  ⚡ Reranking with cross-encoder...");
            Console.ResetColor();

            var candidateChunks  = top5.Select(r => chunks[r.Key]).ToList();
            var candidateIndices = top5.Select(r => r.Key).ToList();

            var rerankedResults = Rerank(
                session, candidateChunks, candidateIndices, query, vocab);

            // ── Step 5: Show retrieved chunks ──────────────────────
            Console.ForegroundColor = ConsoleColor.Magenta;
            Console.WriteLine($"\n  📚 Top 3 sources used:");
            Console.ResetColor();

            var top3 = rerankedResults.Take(3).ToList();
            int rank = 1;
            foreach (var (idx, score) in top3)
            {
                bool   inVector = vectorResults
                    .Any(r => (int)r.Id.Num == idx);
                bool   inBm25   = bm25Results
                    .Any(r => r.index == idx);
                string source   = (inVector && inBm25)
                    ? "Vector + BM25 ⭐"
                    : inVector ? "Vector only" : "BM25 only";

                PrintResult(rank++, score, chunks[idx], source);
            }

            // ── Step 6: Generate answer ────────────────────────────
            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.Cyan;
            Console.WriteLine(
                "  ── GENERATED ANSWER ──────────────────────────────");
            Console.ResetColor();
            Console.ForegroundColor = ConsoleColor.Yellow;
            Console.WriteLine("  ⚡ Mistral is reading the context...\n");
            Console.ResetColor();

            // Extract top 3 chunk texts for context
            // Top 3 Chunk-Texte als Kontext extrahieren
            var contextChunks = top3
                .Select(r => chunks[r.index])
                .ToList();

            Console.ForegroundColor = ConsoleColor.Green;
            Console.Write("  💬 ");
            Console.ResetColor();

            await GenerateAnswer(ollamaClient, query, contextChunks);

            Console.WriteLine();
            Console.ForegroundColor = ConsoleColor.DarkGray;
            Console.WriteLine(
                "  ──────────────────────────────────────────────────");
            Console.ResetColor();
        }

        session.Dispose();

        Console.WriteLine();
        Console.ForegroundColor = ConsoleColor.Cyan;
        Console.WriteLine("  👋 Auf Wiedersehen! Day 5 complete.");
        Console.ResetColor();
        Console.WriteLine();
    }
}
