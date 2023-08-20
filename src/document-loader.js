import { Document } from 'langchain/document';
import { CSVLoader } from 'langchain/document_loaders/fs/csv';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import { JSONLinesLoader, JSONLoader } from 'langchain/document_loaders/fs/json';
import { PDFLoader } from 'langchain/document_loaders/fs/pdf';
import { TextLoader } from 'langchain/document_loaders/fs/text';

async function load() {
  const loader = new TextLoader('assets/example.txt');
  const docs = await loader.load();

  console.log(docs);
}

async function create() {
  const doc = new Document({
    pageContent: 'Hello, World!',
    metadata: {
      source: '1',
    },
  });

  console.log(doc);
}

async function csv() {
  const loader = new CSVLoader('assets/example.csv');

  const docs = await loader.load();

  console.log(docs);
}

async function extractSingleColumn() {
  const loader = new CSVLoader('assets/example.csv', 'text');

  const docs = await loader.load();

  console.log(docs);
}

async function fileDirectory() {
  const loader = new DirectoryLoader('assets', {
    '.json': path => new JSONLoader(path, '/texts'),
    '.jsonl': path => new JSONLinesLoader(path, '/name'),
    '.txt': path => new TextLoader(path),
    '.csv': path => new CSVLoader(path, 'text'),
  });
  const docs = await loader.load();
  console.log(docs);
}

async function pdf() {
  const loader = new PDFLoader('assets/example.pdf');

  const docs = await loader.load();

  console.log(docs);
}
