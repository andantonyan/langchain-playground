import { LLMChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate } from 'langchain/prompts';

const systemMessagePrompt = SystemMessagePromptTemplate.fromTemplate(
  'You are a helpful assistant that translates {input_language} to {output_language}.',
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

// { text: "J'adore programmer" }
