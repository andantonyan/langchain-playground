import { OpenAI } from 'langchain/llms/openai';

import './env.js';

async function llmStreaming() {
  const model = new OpenAI({
    maxTokens: 25,
    streaming: true,
  });

  const response = await model.call('Tell me a joke.', {
    callbacks: [
      {
        handleLLMNewToken(token) {
          console.log({ token });
        },
      },
    ],
  });

  console.log(response);
}
