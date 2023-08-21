import { OpenAIEmbeddings } from 'langchain/embeddings/openai';

import './env.js';

/* Create instance */
const embeddings = new OpenAIEmbeddings();

/* Embed queries */
const res = await embeddings.embedQuery('Hello world');
const documentRes = await embeddings.embedDocuments(['Hello world', 'Bye bye']);
