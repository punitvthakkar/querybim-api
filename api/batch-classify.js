import { createClient } from '@supabase/supabase-js';

// Helper function to get embeddings in batches from the Gemini API
async function getBatchEmbeddings(texts) {
  try {
    const modelName = 'models/text-embedding-004';
    // Construct the request body as per the Gemini API specification
    const requestBody = {
      requests: texts.map(text => ({
        model: modelName,
        content: { parts: [{ text }] }
      }))
    };
    
    const response = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/${modelName}:batchEmbedContents?key=${process.env.GEMINI_API_KEY}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      }
    );

    if (!response.ok) {
      throw new Error(`Gemini API error: ${response.status} ${await response.text()}`);
    }
    
    const data = await response.json();
    // Return an array of embedding vectors, or null for any failures
    return data.embeddings ? data.embeddings.map(emb => emb.values) : new Array(texts.length).fill(null);
  } catch (error) {
    console.error('Batch embedding function error:', error);
    return new Array(texts.length).fill(null);
  }
}

// Main Vercel serverless function handler
export default async function handler(req, res) {
  // Set CORS headers to allow requests from any origin (including the Revit environment)
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle CORS preflight requests
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST requests
  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Method not allowed' });
  }

  try {
    const { queries } = req.body;
    if (!queries || !Array.isArray(queries) || queries.length === 0) {
      return res.status(400).json({ success: false, error: 'Missing or invalid "queries" array in request body' });
    }

    // 1. Get Embeddings from Gemini API
    const queryTexts = queries.map(q => q.query);
    const CHUNK_SIZE = 100; // Gemini batch embedding has a limit of 100 per request
    const embeddingPromises = [];
    for (let i = 0; i < queryTexts.length; i += CHUNK_SIZE) {
      embeddingPromises.push(getBatchEmbeddings(queryTexts.slice(i, i + CHUNK_SIZE)));
    }
    const embeddingChunks = await Promise.all(embeddingPromises);
    const embeddings = embeddingChunks.flat();

    // 2. Prepare data for Supabase RPC call
    const supabase = createClient(process.env.SUPABASE_URL, process.env.SUPABASE_ANON_KEY);
    
    const batch_request_ids = [];
    const batch_embeddings = [];
    const batch_uniclass_types = [];
    const batch_depths = [];
    
    queries.forEach((query, i) => {
        if (embeddings[i]) { // Only include queries that successfully got an embedding
            batch_request_ids.push(Number(query.request_id || i));
            batch_embeddings.push(JSON.stringify(embeddings[i])); // Embeddings must be sent as stringified JSON
            batch_uniclass_types.push(query.uniclass_type.toUpperCase());
            batch_depths.push(Number(query.depth || 2)); // Default to depth 2 if not provided
        }
    });

    // 3. Call the Supabase database function
    const { data: batchData, error: batchError } = await supabase.rpc('querybim_batch_match', {
        p_request_ids: batch_request_ids,
        p_query_embeddings: batch_embeddings,
        p_uniclass_type_filters: batch_uniclass_types,
        p_depths: batch_depths
    });

    if (batchError) {
      console.error('Supabase RPC error:', batchError);
      return res.status(500).json({ success: false, error: 'Database processing failed', details: batchError.message });
    }

    // 4. Format the results and send the response
    const resultsMap = new Map();
    batchData.forEach(item => {
        const result_text = `${item.code}:${item.title}`;
        const scoreFormatted = item.similarity.toFixed(2);
        const finalResult = `${result_text}:${scoreFormatted}`;
        
        resultsMap.set(Number(item.request_id), {
            request_id: Number(item.request_id),
            match: finalResult,
            confidence: item.similarity
        });
    });

    const finalResults = queries.map((query, i) => {
        const requestId = Number(query.request_id || i);
        if (resultsMap.has(requestId)) {
            return resultsMap.get(requestId);
        }
        // Return a structured "no match" or "failed" response
        return {
            request_id: requestId,
            match: embeddings[i] ? 'No match found:0.00' : 'Embedding failed:0.00',
            confidence: 0
        };
    });

    return res.json({ success: true, processed: finalResults.length, results: finalResults });

  } catch (error) {
    console.error('Batch API Handler Error:', error);
    return res.status(500).json({ success: false, error: 'Internal server error' });
  }
}
