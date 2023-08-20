import { RetrievalQAChain, loadQAStuffChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { Document } from 'langchain/document';
import { CheerioWebBaseLoader } from 'langchain/document_loaders/web/cheerio';
import { HtmlToTextTransformer } from 'langchain/document_transformers/html_to_text';
import { MozillaReadabilityTransformer } from 'langchain/document_transformers/mozilla_readability';
import { createMetadataTaggerFromZod } from 'langchain/document_transformers/openai_functions';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import { CharacterTextSplitter, RecursiveCharacterTextSplitter, TokenTextSplitter } from 'langchain/text_splitter';
import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { z } from 'zod';

import './env.js';

async function basic() {
  const text = `Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
  This is a weird text to write, but gotta test the splittingggg some how.\n\n
  Bye!\n\n-H.`;
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 10,
    chunkOverlap: 1,
  });

  const output = await splitter.createDocuments([text]);
  console.log(output);
}

async function splitDocument() {
  const text = `Hi.\n\nI'm Harrison.\n\nHow? Are? You?\nOkay then f f f f.
This is a weird text to write, but gotta test the splittingggg some how.\n\n
Bye!\n\n-H.`;
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 10,
    chunkOverlap: 1,
  });

  const docOutput = await splitter.splitDocuments([new Document({ pageContent: text })]);
  console.log(docOutput);
}

async function htmlToText() {
  const loader = new CheerioWebBaseLoader('https://news.ycombinator.com/item?id=34817881');
  const docs = await loader.load();
  const splitter = RecursiveCharacterTextSplitter.fromLanguage('html');
  const transformer = new HtmlToTextTransformer();
  const sequence = splitter.pipe(transformer);
  const newDocuments = await sequence.invoke(docs);

  console.log(newDocuments);
}

async function readability() {
  const loader = new CheerioWebBaseLoader('https://news.ycombinator.com/item?id=34817881');

  const docs = await loader.load();
  const splitter = RecursiveCharacterTextSplitter.fromLanguage('html');
  const transformer = new MozillaReadabilityTransformer();
  const sequence = splitter.pipe(transformer);
  const newDocuments = await sequence.invoke(docs);

  console.log(newDocuments);
}

async function functionMetadataTrigger() {
  const zodSchema = z.object({
    movie_title: z.string(),
    critic: z.string(),
    tone: z.enum(['positive', 'negative']),
    rating: z.optional(z.number()).describe('The number of stars the critic rated the movie'),
  });

  const metadataTagger = createMetadataTaggerFromZod(zodSchema, {
    llm: new ChatOpenAI({ modelName: 'gpt-3.5-turbo', verbose: true }),
  });

  const documents = [
    new Document({
      pageContent: 'Review of The Bee Movie\nBy Roger Ebert\nThis is the greatest movie ever made. 4 out of 5 stars.',
    }),
    new Document({
      pageContent: 'Review of The Godfather\nBy Anonymous\n\nThis movie was super boring. 1 out of 5 stars.',
      metadata: { reliable: false },
    }),
  ];
  const taggedDocuments = await metadataTagger.transformDocuments(documents);

  console.log(taggedDocuments);
}

async function splitByChar() {
  const text = 'foo bar baz 123';
  const splitter = new CharacterTextSplitter({
    separator: ' ',
    chunkSize: 7,
    chunkOverlap: 3,
  });
  const output = await splitter.createDocuments([text]);
  console.log(output);
}

async function splitCode() {
  const jsCode = `function helloWorld() {
  console.log("Hello, World!");
}
// Call the function
helloWorld();`;

  const splitter = RecursiveCharacterTextSplitter.fromLanguage('js', {
    chunkSize: 32,
    chunkOverlap: 0,
  });
  const jsOutput = await splitter.createDocuments([jsCode]);

  console.log(jsOutput);
}

async function contextualChunkHeaders() {
  const splitter = new CharacterTextSplitter({
    chunkSize: 1536,
    chunkOverlap: 200,
  });

  const jimDocs = await splitter.createDocuments([`My favorite color is blue.`], [], {
    chunkHeader: `DOCUMENT NAME: Jim Interview\n\n---\n\n`,
    appendChunkOverlapHeader: true,
  });

  const pamDocs = await splitter.createDocuments([`My favorite color is red.`], [], {
    chunkHeader: `DOCUMENT NAME: Pam Interview\n\n---\n\n`,
    appendChunkOverlapHeader: true,
  });

  const vectorStore = await HNSWLib.fromDocuments(jimDocs.concat(pamDocs), new OpenAIEmbeddings());

  const model = new OpenAI({ temperature: 0 });

  const chain = new RetrievalQAChain({
    combineDocumentsChain: loadQAStuffChain(model),
    retriever: vectorStore.asRetriever(),
    returnSourceDocuments: true,
  });
  const res = await chain.call({
    query: "What is Pam's favorite color?",
  });

  console.log(JSON.stringify(res, null, 2));
}

async function tokenTextSplitter() {
  const text = 'foo bar baz 123';

  const splitter = new TokenTextSplitter({
    encodingName: 'gpt2',
    chunkSize: 10,
    chunkOverlap: 0,
  });

  const output = await splitter.createDocuments([text]);
  console.log(output);
}
