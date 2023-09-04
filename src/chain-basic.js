import { ConversationChain, LLMChain, SequentialChain, SimpleSequentialChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { OpenAI } from 'langchain/llms/openai';
import { BufferMemory } from 'langchain/memory';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from 'langchain/prompts';

import './env.js';

async function basic() {
  // We can construct an LLMChain from a PromptTemplate and an LLM.
  const model = new OpenAI({ temperature: 0 });
  const prompt = PromptTemplate.fromTemplate('What is a good name for a company that makes {product}?');

  const chain = new LLMChain({ llm: model, prompt });

  // Since this LLMChain is a single-input, single-output chain, we can also `run` it.
  // This convenience method takes in a string and returns the value
  // of the output key field in the chain response. For LLMChains, this defaults to "text".
  const res = await chain.run('colorful socks');
  console.log(res);
}

async function basic2() {
  const model = new OpenAI({ temperature: 0 });
  const prompt = PromptTemplate.fromTemplate('What is a good name for {company} that makes {product}?');

  const chain = new LLMChain({ llm: model, prompt });

  const res = await chain.call({
    company: 'a startup',
    product: 'colorful socks',
  });
  console.log(res);
}

async function chatBasic() {
  const chat = new ChatOpenAI({ temperature: 0 });
  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      'You are a helpful assistant that translates {input_language} to {output_language}.',
    ),
    HumanMessagePromptTemplate.fromTemplate('{text}'),
  ]);
  const chain = new LLMChain({
    prompt: chatPrompt,
    llm: chat,
  });

  const res = await chain.call({
    input_language: 'English',
    output_language: 'French',
    text: 'I love programming.',
  });
  console.log(res);
}

async function stateful() {
  const chat = new ChatOpenAI({});

  const memory = new BufferMemory();

  // This particular chain automatically initializes a BufferMemory instance if none is provided,
  // but we pass it explicitly here. It also has a default prompt.
  const chain = new ConversationChain({ llm: chat, memory });

  const res1 = await chain.run('Answer briefly. What are the first 3 colors of a rainbow?');
  console.log(res1);

  // The first three colors of a rainbow are red, orange, and yellow.

  const res2 = await chain.run('And the next 4?');
  console.log(res2);
  // The next four colors of a rainbow are green, blue, indigo, and violet.
}

async function simpleSequentialChain() {
  // This is an LLMChain to write a synopsis given a title of a play.
  const llm = new OpenAI({ temperature: 0 });
  const template = `You are a playwright. Given the title of play, it is your job to write a synopsis for that title.
 
  Title: {title}
  Playwright: This is a synopsis for the above play:`;
  const promptTemplate = new PromptTemplate({
    template,
    inputVariables: ['title'],
  });
  const synopsisChain = new LLMChain({ llm, prompt: promptTemplate });

  // This is an LLMChain to write a review of a play given a synopsis.
  const reviewLLM = new OpenAI({ temperature: 0 });
  const reviewTemplate = `You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
 
  Play Synopsis:
  {synopsis}
  Review from a New York Times play critic of the above play:`;
  const reviewPromptTemplate = new PromptTemplate({
    template: reviewTemplate,
    inputVariables: ['synopsis'],
  });
  const reviewChain = new LLMChain({
    llm: reviewLLM,
    prompt: reviewPromptTemplate,
  });

  const overallChain = new SimpleSequentialChain({
    chains: [synopsisChain, reviewChain],
    verbose: true,
  });
  const review = await overallChain.run('Tragedy at sunset on the beach');
  console.log(review);
}

async function sequentialChain() {
  // This is an LLMChain to write a synopsis given a title of a play and the era it is set in.
  const llm = new OpenAI({ temperature: 0 });
  const template = `You are a playwright. Given the title of play and the era it is set in, it is your job to write a synopsis for that title.

Title: {title}
Era: {era}
Playwright: This is a synopsis for the above play:`;
  const promptTemplate = new PromptTemplate({
    template,
    inputVariables: ['title', 'era'],
  });
  const synopsisChain = new LLMChain({
    llm,
    prompt: promptTemplate,
    outputKey: 'synopsis',
  });

  // This is an LLMChain to write a review of a play given a synopsis.
  const reviewLLM = new OpenAI({ temperature: 0 });
  const reviewTemplate = `You are a play critic from the New York Times. Given the synopsis of play, it is your job to write a review for that play.
  
   Play Synopsis:
   {synopsis}
   Review from a New York Times play critic of the above play:`;
  const reviewPromptTemplate = new PromptTemplate({
    template: reviewTemplate,
    inputVariables: ['synopsis'],
  });
  const reviewChain = new LLMChain({
    llm: reviewLLM,
    prompt: reviewPromptTemplate,
    outputKey: 'review',
  });

  const overallChain = new SequentialChain({
    chains: [synopsisChain, reviewChain],
    inputVariables: ['era', 'title'],
    // Here we return multiple variables
    outputVariables: ['synopsis', 'review'],
    verbose: true,
  });
  const chainExecutionResult = await overallChain.call({
    title: 'Tragedy at sunset on the beach',
    era: 'Victorian England',
  });
  console.log(chainExecutionResult);
}
