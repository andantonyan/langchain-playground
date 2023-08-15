import { initializeAgentExecutorWithOptions } from 'langchain/agents';
import { ConversationChain, LLMChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { OpenAI } from 'langchain/llms/openai';
import { BufferMemory } from 'langchain/memory';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from 'langchain/prompts';
import { HumanMessage } from 'langchain/schema';
import { SerpAPI } from 'langchain/tools';
import { Calculator } from 'langchain/tools/calculator';

import './env.js';

async function predictUsingLLM() {
  const llm = new OpenAI({
    temperature: 0.9,
  });

  const result = await llm.predict('What would be a good company name for a company that makes colorful socks?');
  console.log(`Company name: ${result}`);
}

async function translateUsingChat() {
  const chat = new ChatOpenAI({
    temperature: 0,
  });

  const result = await chat.predictMessages([
    new HumanMessage('Translate this sentence from English to French. I love programming.'),
  ]);

  console.log('result', JSON.stringify(result, null, 2));
}

async function promptTemplate() {
  const prompt = PromptTemplate.fromTemplate('What is a good name for a company that makes {product}?');

  const llmFormattedPrompt = await prompt.format({
    product: 'colorful socks',
  });

  console.log(`llmFormattedPrompt: ${llmFormattedPrompt}`);

  const systemMessagePrompt = SystemMessagePromptTemplate.fromTemplate(
    'You are a helpful assistant that translates {input_language} to {output_language}.'
  );
  const humanMessagePrompt = HumanMessagePromptTemplate.fromTemplate({ text });

  const chatPrompt = ChatPromptTemplate.fromPromptMessages([systemMessagePrompt, humanMessagePrompt]);

  const modelFormattedPrompt = await chatPrompt.formatMessages({
    input_language: 'English',
    output_language: 'French',
    text: 'I love programming.',
  });

  console.log(`modelFormattedPrompt: ${modelFormattedPrompt}`);
}

async function llsChains() {
  const llm = new OpenAI({});
  const prompt = PromptTemplate.fromTemplate('What is a good name for a company that makes {product}?');

  const chain = new LLMChain({
    llm,
    prompt,
  });

  const result = await chain.run('colorful socks');

  console.log('result', JSON.stringify(result, null, 2));
}

async function chatChains() {
  const systemMessagePrompt = SystemMessagePromptTemplate.fromTemplate(
    'You are a helpful assistant that translates {input_language} to {output_language}.'
  );
  const humanMessagePrompt = HumanMessagePromptTemplate.fromTemplate('{text}');
  const chatPrompt = ChatPromptTemplate.fromPromptMessages([systemMessagePrompt, humanMessagePrompt]);

  const chat = new ChatOpenAI({
    temperature: 0,
  });

  const chain = new LLMChain({
    llm: chat,
    prompt: chatPrompt,
  });

  const result = await chain.call({
    input_language: 'English',
    output_language: 'French',
    text: 'I love programming',
  });

  console.log('result', JSON.stringify(result, null, 2));
}

async function llmAgents() {
  const model = new OpenAI({ temperature: 0 });
  const tools = [
    new SerpAPI(process.env.SERPAPI_API_KEY, {
      location: 'Austin,Texas,United States',
      hl: 'en',
      gl: 'us',
    }),
    new Calculator(),
  ];

  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: 'zero-shot-react-description',
    verbose: true,
  });

  const input =
    'What was the high temperature in SF yesterday in Fahrenheit? What is that number raised to the .023 power?';

  const result = await executor.call({ input });

  console.log('result', JSON.stringify(result, null, 2));
}

async function chatAgents() {
  const executor = await initializeAgentExecutorWithOptions(
    [new Calculator(), new SerpAPI()],
    new ChatOpenAI({ modelName: 'gpt-3.5-turbo', temperature: 0 }),
    {
      agentType: 'openai-functions',
      verbose: true,
    }
  );

  const result = await executor.run('What is the temperature in New York?');

  console.log('result', JSON.stringify(result, null, 2));
}

async function llmMemory() {
  const model = new OpenAI({});
  const memory = new BufferMemory();
  const chain = new ConversationChain({
    llm: model,
    memory,
    verbose: true,
  });

  const res1 = await chain.call({ input: "Hi! I'm Jim." });
  const res2 = await chain.call({ input: "What's my name?" });
}
