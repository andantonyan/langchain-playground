import * as fs from 'fs';
import { RetrievalQAChain } from 'langchain/chains';
import { Document } from 'langchain/document';
import { TextLoader } from 'langchain/document_loaders/fs/text';
import { CacheBackedEmbeddings } from 'langchain/embeddings/cache_backed';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import { ContextualCompressionRetriever } from 'langchain/retrievers/contextual_compression';
import { LLMChainExtractor } from 'langchain/retrievers/document_compressors/chain_extract';
import { SelfQueryRetriever } from 'langchain/retrievers/self_query';
import { FunctionalTranslator } from 'langchain/retrievers/self_query/functional';
import { TimeWeightedVectorStoreRetriever } from 'langchain/retrievers/time_weighted';
import { InMemoryStore } from 'langchain/storage/in_memory';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { FaissStore } from 'langchain/vectorstores/faiss';
import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

import './env.js';

async function basic() {
  const vectorStore = await MemoryVectorStore.fromTexts(
    ['Hello world', 'Bye bye', 'hello nice world'],
    [{ id: 2 }, { id: 1 }, { id: 3 }],
    new OpenAIEmbeddings(),
  );

  const resultOne = await vectorStore.similaritySearch('hello world', 1);
}

async function contextualCompression() {
  const model = new OpenAI({ verbose: true });
  const baseCompressor = LLMChainExtractor.fromLLM(model);

  const text = fs.readFileSync('assets/state_of_the_union.txt', 'utf8');
  const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 500 });
  const docs = await textSplitter.createDocuments([text]);

  const vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
  const retriever = new ContextualCompressionRetriever({
    baseCompressor,
    baseRetriever: vectorStore.asRetriever(),
  });

  const chain = RetrievalQAChain.fromLLM(model, retriever);
  const res = await chain.call({ query: 'What is key topics?' });
  console.log({ res });
}

async function selfQuerying() {
  /**
   * First, we create a bunch of documents. You can load your own documents here instead.
   * Each document has a pageContent and a metadata field. Make sure your metadata matches the AttributeInfo below.
   */
  const docs = [
    new Document({
      pageContent: 'A bunch of scientists bring back dinosaurs and mayhem breaks loose',
      metadata: { year: 1993, rating: 7.7, genre: 'science fiction' },
    }),
    new Document({
      pageContent: 'Leo DiCaprio gets lost in a dream within a dream within a dream within a ...',
      metadata: { year: 2010, director: 'Christopher Nolan', rating: 8.2 },
    }),
    new Document({
      pageContent:
        'A psychologist / detective gets lost in a series of dreams within dreams within dreams and Inception reused the idea',
      metadata: { year: 2006, director: 'Satoshi Kon', rating: 8.6 },
    }),
    new Document({
      pageContent: 'A bunch of normal-sized women are supremely wholesome and some men pine after them',
      metadata: { year: 2019, director: 'Greta Gerwig', rating: 8.3 },
    }),
    new Document({
      pageContent: 'Toys come alive and have a blast doing so',
      metadata: { year: 1995, genre: 'animated' },
    }),
    new Document({
      pageContent: 'Three men walk into the Zone, three men walk out of the Zone',
      metadata: {
        year: 1979,
        director: 'Andrei Tarkovsky',
        genre: 'science fiction',
        rating: 9.9,
      },
    }),
  ];

  /**
   * Next, we define the attributes we want to be able to query on.
   * in this case, we want to be able to query on the genre, year, director, rating, and length of the movie.
   * We also provide a description of each attribute and the type of the attribute.
   * This is used to generate the query prompts.
   */
  const attributeInfo = [
    {
      name: 'genre',
      description: 'The genre of the movie',
      type: 'string or array of strings',
    },
    {
      name: 'year',
      description: 'The year the movie was released',
      type: 'number',
    },
    {
      name: 'director',
      description: 'The director of the movie',
      type: 'string',
    },
    {
      name: 'rating',
      description: 'The rating of the movie (1-10)',
      type: 'number',
    },
    {
      name: 'length',
      description: 'The length of the movie in minutes',
      type: 'number',
    },
  ];

  /**
   * Next, we instantiate a vector store. This is where we store the embeddings of the documents.
   * We also need to provide an embeddings object. This is used to embed the documents.
   */
  const embeddings = new OpenAIEmbeddings();
  const llm = new OpenAI();
  const documentContents = 'Brief summary of a movie';
  const vectorStore = await MemoryVectorStore.fromDocuments(docs, embeddings);
  const selfQueryRetriever = await SelfQueryRetriever.fromLLM({
    llm,
    vectorStore,
    documentContents,
    attributeInfo,
    /**
     * We need to use a translator that translates the queries into a
     * filter format that the vector store can understand. We provide a basic translator
     * here, but you can create your own translator by extending BaseTranslator
     * abstract class. Note that the vector store needs to support filtering on the metadata
     * attributes you want to query on.
     */
    structuredQueryTranslator: new FunctionalTranslator(),
  });

  /**
   * Now we can query the vector store.
   * We can ask questions like "Which movies are less than 90 minutes?" or "Which movies are rated higher than 8.5?".
   * We can also ask questions like "Which movies are either comedy or drama and are less than 90 minutes?".
   * The retriever will automatically convert these questions into queries that can be used to retrieve documents.
   */
  const query1 = await selfQueryRetriever.getRelevantDocuments('Which movies are less than 90 minutes?');
  const query2 = await selfQueryRetriever.getRelevantDocuments('Which movies are rated higher than 8.5?');
  const query3 = await selfQueryRetriever.getRelevantDocuments('Which movies are directed by Greta Gerwig?');
  const query4 = await selfQueryRetriever.getRelevantDocuments(
    'Which movies are either comedy or drama and are less than 90 minutes?',
  );
  console.log(query1, query2, query3, query4);
}

async function timeWeightedRetriever() {
  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());

  const retriever = new TimeWeightedVectorStoreRetriever({
    vectorStore,
    memoryStream: [],
    searchKwargs: 2,
  });

  const documents = [
    'My name is John.',
    'My name is Bob.',
    'My favourite food is pizza.',
    'My favourite food is pasta.',
    'My favourite food is sushi.',
  ].map(pageContent => ({ pageContent, metadata: {} }));

  // All documents must be added using this method on the retriever (not the vector store!)
  // so that the correct access history metadata is populated
  await retriever.addDocuments(documents);

  const results1 = await retriever.getRelevantDocuments('What is my favourite food?');

  console.log(results1);

  /*
  [
    Document { pageContent: 'My favourite food is pasta.', metadata: {} }
  ]
   */

  const results2 = await retriever.getRelevantDocuments('What is my favourite food?');

  console.log(results2);

  /*
  [
    Document { pageContent: 'My favourite food is pasta.', metadata: {} }
  ]
   */
}

async function cachingEmbeddings() {
  const underlyingEmbeddings = new OpenAIEmbeddings();

  const inMemoryStore = new InMemoryStore();

  const cacheBackedEmbeddings = CacheBackedEmbeddings.fromBytesStore(underlyingEmbeddings, inMemoryStore, {
    namespace: underlyingEmbeddings.modelName,
  });

  const loader = new TextLoader('assets/state_of_the_union.txt');
  const rawDocuments = await loader.load();
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 1000,
    chunkOverlap: 0,
  });
  const documents = await splitter.splitDocuments(rawDocuments);

  // No keys logged yet since the cache is empty
  for await (const key of inMemoryStore.yieldKeys()) {
    console.log(key);
  }

  let time = Date.now();
  const vectorstore = await FaissStore.fromDocuments(documents, cacheBackedEmbeddings);
  console.log(`Initial creation time: ${Date.now() - time}ms`);
  /*
    Initial creation time: 1905ms
  */

  // The second time is much faster since the embeddings for the input docs have already been added to the cache
  time = Date.now();
  const vectorstore2 = await FaissStore.fromDocuments(documents, cacheBackedEmbeddings);
  console.log(`Cached creation time: ${Date.now() - time}ms`);
  /*
    Cached creation time: 8ms
  */

  // Many keys logged with hashed values
  const keys = [];
  for await (const key of inMemoryStore.yieldKeys()) {
    keys.push(key);
  }

  console.log(keys.slice(0, 5));
  /*
    [
      'text-embedding-ada-002ea9b59e760e64bec6ee9097b5a06b0d91cb3ab64',
      'text-embedding-ada-0023b424f5ed1271a6f5601add17c1b58b7c992772e',
      'text-embedding-ada-002fec5d021611e1527297c5e8f485876ea82dcb111',
      'text-embedding-ada-00262f72e0c2d711c6b861714ee624b28af639fdb13',
      'text-embedding-ada-00262d58882330038a4e6e25ea69a938f4391541874'
    ]
  */
}
