import { OpenAI } from 'langchain/llms/openai';
import { PromptTemplate } from 'langchain/prompts';

import './env.js';

async function tellMeAJoke() {
  const llm = new OpenAI({});
  const prompt = PromptTemplate.fromTemplate('Tell me a {adjective} joke');

  const joke = await llm.call(await prompt.format({ adjective: 'funny' }));
  console.log(joke);
}

async function batch() {
  const llm = new OpenAI({});
  const llmResult = await llm.generate(['Tell me a joke', 'Tell me a poem'], ['Tell me a joke', 'Tell me a poem']);
  console.log(llmResult.generations.length);
  console.log(llmResult.generations[0]);
}

async function generateCompanyName() {
  const model = new OpenAI({
    // customize openai model that's used, `text-davinci-003` is the default
    modelName: 'text-ada-001',

    // `max_tokens` supports a magic -1 param where the max token length for the specified modelName
    //  is calculated and included in the request to OpenAI as the `max_tokens` param
    maxTokens: -1,

    // use `modelKwargs` to pass params directly to the openai call
    // note that they use snake_case instead of camelCase
    modelKwargs: {
      user: 'me',
    },

    // for additional logging for debugging purposes
    verbose: true,
  });

  const resA = await model.call('What would be a good company name a company that makes colorful socks?');
  console.log({ resA });
  // { resA: '\n\nSocktastic Colors' }
}

async function cancelingRequests() {
  const llm = new OpenAI({});
  const prompt = PromptTemplate.fromTemplate('Tell me a {adjective} joke');
  const abortController = new AbortController();
  const formattedPrompt = await prompt.format({ adjective: 'funny' });
  setTimeout(() => abortController.abort(), 100);
  const joke = await llm.call(formattedPrompt, { signal: abortController.signal });
  console.log(joke);
}
