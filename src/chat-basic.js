import { ChatOpenAI } from 'langchain/chat_models/openai';
import { HumanMessage, SystemMessage } from 'langchain/schema';

import './env.js';

async function generateCompanyName() {
  const model = new ChatOpenAI();

  const res = await model.call([new HumanMessage('What is a good name for a company that makes colorful socks?')]);

  console.log(res);

  const res2 = await model.call([
    new SystemMessage('You are a helpful assistant that translates English to French.'),
    new HumanMessage('Translate: I love programming.'),
  ]);

  console.log(res2);

  const res3 = await model.generate([
    [
      new SystemMessage('You are a helpful assistant that translates English to French.'),
      new HumanMessage('Translate this sentence from English to French. I love programming.'),
    ],
    [
      new SystemMessage('You are a helpful assistant that translates English to French.'),
      new HumanMessage('Translate this sentence from English to French. I love artificial intelligence.'),
    ],
  ]);
  console.log(res3);
}
