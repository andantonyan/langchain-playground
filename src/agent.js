import { initializeAgentExecutorWithOptions } from 'langchain/agents';
import { ChatAnthropic } from 'langchain/chat_models';
import { ChatOpenAI } from 'langchain/chat_models/openai';
import { PlanAndExecuteAgentExecutor } from 'langchain/experimental/plan_and_execute';
import { OpenAI } from 'langchain/llms/openai';
import { DynamicStructuredTool, SerpAPI } from 'langchain/tools';
import { Calculator } from 'langchain/tools/calculator';

import './env.js';

async function weather() {
  const tools = [new Calculator(), new SerpAPI()];
  const chat = new ChatOpenAI({ modelName: 'gpt-4', temperature: 0 });

  const executor = await initializeAgentExecutorWithOptions(tools, chat, {
    agentType: 'openai-functions',
    verbose: true,
  });

  const result = await executor.run('What is the weather in New York?');
  console.log(result);
}

async function conversation() {
  process.env.LANGCHAIN_HANDLER = 'langchain';
  const model = new ChatOpenAI({ temperature: 0 });
  const tools = [
    new SerpAPI(process.env.SERPAPI_API_KEY, {
      location: 'Austin,Texas,United States',
      hl: 'en',
      gl: 'us',
    }),
    new Calculator(),
  ];

  // Passing "chat-conversational-react-description" as the agent type
  // automatically creates and uses BufferMemory with the executor.
  // If you would like to override this, you can pass in a custom
  // memory option, but the memoryKey set on it must be "chat_history".
  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: 'chat-conversational-react-description',
    verbose: true,
  });
  console.log('Loaded agent.');

  const input0 = 'hi, i am bob';

  const result0 = await executor.call({ input: input0 });

  console.log(`Got output ${result0.output}`);

  const input1 = 'whats my name?';

  const result1 = await executor.call({ input: input1 });

  console.log(`Got output ${result1.output}`);

  const input2 = 'whats the weather in pomfret?';

  const result2 = await executor.call({ input: input2 });

  console.log(`Got output ${result2.output}`);
}

async function openAIFunctions() {
  const tools = [new Calculator(), new SerpAPI()];
  const chat = new ChatOpenAI({ modelName: 'gpt-4', temperature: 0 });

  const executor = await initializeAgentExecutorWithOptions(tools, chat, {
    agentType: 'openai-functions',
    verbose: true,
  });

  const result = await executor.run('What is the weather in New York?');
  console.log(result);
}

async function promptCustomization() {
  const tools = [new Calculator(), new SerpAPI()];
  const chat = new ChatOpenAI({ modelName: 'gpt-4', temperature: 0 });
  const prefix = 'You are a helpful AI assistant. However, all final response to the user must be in pirate dialect.';

  const executor = await initializeAgentExecutorWithOptions(tools, chat, {
    agentType: 'openai-functions',
    verbose: true,
    agentArgs: {
      prefix,
    },
  });

  const result = await executor.run('What is the weather in New York?');
  console.log(result);
}

async function planAndExecute() {
  const tools = [new Calculator(), new SerpAPI()];
  const model = new ChatOpenAI({
    temperature: 0,
    modelName: 'gpt-3.5-turbo',
    verbose: true,
  });
  const executor = PlanAndExecuteAgentExecutor.fromLLMAndTools({
    llm: model,
    tools,
  });

  const result = await executor.call({
    input: `Who is the current president of the United States? What is their current age raised to the second power?`,
  });

  console.log({ result });
}

async function reAct() {
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

  const input = `Who is Olivia Wilde's boyfriend? What is his current age raised to the 0.23 power?`;

  const result = await executor.call({ input });
}

async function structuredToolChat() {
  const model = new ChatOpenAI({ temperature: 0 });
  const tools = [
    new Calculator(), // Older existing single input tools will still work
    new DynamicStructuredTool({
      name: 'random-number-generator',
      description: 'generates a random number between two input numbers',
      schema: z.object({
        low: z.number().describe('The lower bound of the generated number'),
        high: z.number().describe('The upper bound of the generated number'),
      }),
      func: async ({ low, high }) => (Math.random() * (high - low) + low).toString(), // Outputs still must be strings
      returnDirect: false, // This is an option that allows the tool to return the output directly
    }),
  ];

  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: 'structured-chat-zero-shot-react-description',
    verbose: true,
  });
  console.log('Loaded agent.');

  const input = `What is a random number between 5 and 10 raised to the second power?`;

  console.log(`Executing with input "${input}"...`);

  const result = await executor.call({ input });

  console.log({ result });
}

async function xmlAgent() {
  const model = new ChatAnthropic({ modelName: 'claude-2', temperature: 0.1 });
  const tools = [new SerpAPI()];

  const executor = await initializeAgentExecutorWithOptions(tools, model, {
    agentType: 'xml',
    verbose: true,
  });
  console.log('Loaded agent.');

  const input = `What is the weather in Honolulu?`;

  const result = await executor.call({ input });

  console.log(result);
}
