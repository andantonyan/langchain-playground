import { ConversationChain, LLMChain } from 'langchain/chains';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { OpenAI } from 'langchain/llms/openai';
import {
  BufferMemory,
  BufferWindowMemory,
  ChatMessageHistory,
  CombinedMemory,
  ConversationSummaryBufferMemory,
  ConversationSummaryMemory,
  ENTITY_MEMORY_CONVERSATION_TEMPLATE,
  EntityMemory,
  VectorStoreRetrieverMemory,
} from 'langchain/memory';
import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  MessagesPlaceholder,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from 'langchain/prompts';
import { AIMessage, HumanMessage } from 'langchain/schema';
import { MemoryVectorStore } from 'langchain/vectorstores/memory';

import './env.js';

async function chatHistory() {
  const history = new ChatMessageHistory();

  await history.addUserMessage('Hi!');

  await history.addAIChatMessage("What's up?");

  const messages = await history.getMessages();

  console.log(messages);

  const pastMessages = [new HumanMessage("My name's Jonas"), new AIMessage('Nice to meet you, Jonas!')];

  const memory = new BufferMemory({
    chatHistory: new ChatMessageHistory(pastMessages),
  });
}

async function bufferMemory() {
  const model = new OpenAI({});
  const memory = new BufferMemory();
  // This chain is preconfigured with a default prompt
  const chain = new ConversationChain({ llm: model, memory: memory });
  const res1 = await chain.call({ input: "Hi! I'm Jim." });
  console.log({ res1 });

  const res2 = await chain.call({ input: "What's my name?" });
  console.log({ res2 });
}

async function chatMemory() {
  const chat = new ChatOpenAI({ temperature: 0 });

  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      'The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.',
    ),
    new MessagesPlaceholder('history'),
    HumanMessagePromptTemplate.fromTemplate('{input}'),
  ]);

  const chain = new ConversationChain({
    memory: new BufferMemory({ returnMessages: true, memoryKey: 'history' }),
    prompt: chatPrompt,
    llm: chat,
  });

  const response = await chain.call({
    input: 'hi! whats up?',
  });

  console.log(response);
}

async function conversationBufferWindowMemory() {
  const model = new OpenAI({});
  const memory = new BufferWindowMemory({ k: 1 });
  const chain = new ConversationChain({ llm: model, memory: memory });
  const res1 = await chain.call({ input: "Hi! I'm Jim." });
  console.log({ res1 });
}

async function entityMemory() {
  const memory = new EntityMemory({
    llm: new OpenAI({ temperature: 0 }),
    chatHistoryKey: 'history', // Default value
    entitiesKey: 'entities', // Default value
  });
  const model = new OpenAI({ temperature: 0.9 });
  const chain = new LLMChain({
    llm: model,
    prompt: ENTITY_MEMORY_CONVERSATION_TEMPLATE, // Default prompt - must include the set chatHistoryKey and entitiesKey as input variables.
    memory,
  });

  const res1 = await chain.call({ input: "Hi! I'm Jim." });
  console.log({
    res1,
    memory: await memory.loadMemoryVariables({ input: 'Who is Jim?' }),
  });

  const res2 = await chain.call({
    input: 'I work in construction. What about you?',
  });
  console.log({
    res2,
    memory: await memory.loadMemoryVariables({ input: 'Who is Jim?' }),
  });
}

async function multipleMemory() {
  const bufferMemory = new BufferMemory({
    memoryKey: 'chat_history_lines',
    inputKey: 'input',
  });

  const summaryMemory = new ConversationSummaryMemory({
    llm: new ChatOpenAI({ modelName: 'gpt-3.5-turbo', temperature: 0 }),
    inputKey: 'input',
    memoryKey: 'conversation_summary',
  });

  const memory = new CombinedMemory({
    memories: [bufferMemory, summaryMemory],
  });

  const _DEFAULT_TEMPLATE = `The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.
  
  Summary of conversation:
  {conversation_summary}
  Current conversation:
  {chat_history_lines}
  Human: {input}
  AI:`;

  const prompt = new PromptTemplate({
    inputVariables: ['input', 'conversation_summary', 'chat_history_lines'],
    template: _DEFAULT_TEMPLATE,
  });
  const model = new ChatOpenAI({ temperature: 0.9, verbose: true });
  const chain = new ConversationChain({ llm: model, memory, prompt });

  const res1 = await chain.call({ input: "Hi! I'm Jim." });
  console.log({ res1 });

  const res2 = await chain.call({ input: 'Can you tell me a joke?' });
  console.log({ res2 });

  const res3 = await chain.call({
    input: "What's my name and what joke did you just tell?",
  });
  console.log({ res3 });
}

async function conversationSummaryMemory() {
  const memory = new ConversationSummaryMemory({
    memoryKey: 'chat_history',
    llm: new OpenAI({ modelName: 'gpt-3.5-turbo', temperature: 0 }),
  });

  const model = new OpenAI({ temperature: 0.9 });
  const prompt =
    PromptTemplate.fromTemplate(`The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

  Current conversation:
  {chat_history}
  Human: {input}
  AI:`);
  const chain = new LLMChain({ llm: model, prompt, memory });

  const res1 = await chain.call({ input: "Hi! I'm Jim." });
  console.log({ res1, memory: await memory.loadMemoryVariables({}) });

  const res2 = await chain.call({ input: "What's my name?" });
  console.log({ res2, memory: await memory.loadMemoryVariables({}) });
}

async function conversationSummaryBufferMemory() {
  const memory = new ConversationSummaryBufferMemory({
    llm: new OpenAI({ modelName: 'text-davinci-003', temperature: 0 }),
    maxTokenLimit: 10,
  });

  await memory.saveContext({ input: 'hi' }, { output: 'whats up' });
  await memory.saveContext({ input: 'not much you' }, { output: 'not much' });
  const history = await memory.loadMemoryVariables({});
  console.log({ history });

  const chatPromptMemory = new ConversationSummaryBufferMemory({
    llm: new ChatOpenAI({ modelName: 'gpt-3.5-turbo', temperature: 0 }),
    maxTokenLimit: 10,
    returnMessages: true,
  });
  await chatPromptMemory.saveContext({ input: 'hi' }, { output: 'whats up' });
  await chatPromptMemory.saveContext({ input: 'not much you' }, { output: 'not much' });

  const messages = await chatPromptMemory.chatHistory.getMessages();
  const previous_summary = '';
  const predictSummary = await chatPromptMemory.predictNewSummary(messages, previous_summary);
  console.log(JSON.stringify(predictSummary));

  const chatPrompt = ChatPromptTemplate.fromPromptMessages([
    SystemMessagePromptTemplate.fromTemplate(
      'The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.',
    ),
    new MessagesPlaceholder('history'),
    HumanMessagePromptTemplate.fromTemplate('{input}'),
  ]);

  const model = new ChatOpenAI({ temperature: 0.9, verbose: true });
  const chain = new ConversationChain({
    llm: model,
    memory: chatPromptMemory,
    prompt: chatPrompt,
  });

  const res1 = await chain.predict({ input: "Hi, what's up?" });
  console.log({ res1 });

  const res2 = await chain.predict({
    input: 'Just working on writing some documentation!',
  });
  console.log({ res2 });

  const res3 = await chain.predict({
    input: 'For LangChain! Have you heard of it?',
  });
  console.log({ res3 });

  const res4 = await chain.predict({
    input: "That's not the right one, although a lot of people confuse it for that!",
  });
  console.log({ res4 });
}

async function vectorStoreMemory() {
  const vectorStore = new MemoryVectorStore(new OpenAIEmbeddings());
  const memory = new VectorStoreRetrieverMemory({
    vectorStoreRetriever: vectorStore.asRetriever(1),
    memoryKey: 'history',
  });

  await memory.saveContext({ input: 'My favorite food is pizza' }, { output: 'thats good to know' });
  await memory.saveContext({ input: 'My favorite sport is soccer' }, { output: '...' });
  await memory.saveContext({ input: "I don't the Celtics" }, { output: 'ok' });

  console.log(await memory.loadMemoryVariables({ prompt: 'what sport should i watch?' }));

  const model = new OpenAI({ temperature: 0.9 });
  const prompt =
    PromptTemplate.fromTemplate(`The following is a friendly conversation between a human and an AI. The AI is talkative and provides lots of specific details from its context. If the AI does not know the answer to a question, it truthfully says it does not know.

Relevant pieces of previous conversation:
{history}

(You do not need to use these pieces of information if not relevant)

Current conversation:
Human: {input}
AI:`);
  const chain = new LLMChain({ llm: model, prompt, memory });

  const res1 = await chain.call({ input: "Hi, my name is Perry, what's up?" });
  console.log({ res1 });

  const res2 = await chain.call({ input: "what's my favorite sport?" });
  console.log({ res2 });

  const res3 = await chain.call({ input: "what's my name?" });
  console.log({ res3 });
}
