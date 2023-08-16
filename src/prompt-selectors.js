import { LLMChain, StuffDocumentsChain } from 'langchain/chains';
import { OpenAI } from 'langchain/llms/openai';
import {
  ChatPromptTemplate,
  ConditionalPromptSelector,
  HumanMessagePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
  isChatModel,
} from 'langchain/prompts';

import './env.js';

const openAI = new OpenAI({});
const DEFAULT_QA_PROMPT = PromptTemplate.fromTemplate(
  'Translate this sentence from English to French. I love programming.',
);
const CHAT_PROMPT = ChatPromptTemplate.fromPromptMessages([
  SystemMessagePromptTemplate.fromTemplate(
    'You are a helpful assistant that translates {input_language} to {output_language}.',
  ),
  HumanMessagePromptTemplate.fromTemplate('{text}'),
]);

const QA_PROMPT_SELECTOR = new ConditionalPromptSelector(DEFAULT_QA_PROMPT, [[isChatModel, CHAT_PROMPT]]);

async function generateFormattedPrompt(llm) {
  const promptTemplate = await QA_PROMPT_SELECTOR.getPrompt(llm);
  const prompt = await promptTemplate.format({
    text: 'I love programming.',
    input_language: 'English',
    output_language: 'French',
  });
  console.log(prompt);
}

await generateFormattedPrompt(openAI);

function loadQAStuffChain(llm, params = {}) {
  const { prompt = QA_PROMPT_SELECTOR.getPrompt(llm) } = params;
  const llmChain = new LLMChain({ prompt, llm });
  return new StuffDocumentsChain({ llmChain });
}
