import { ChatOpenAI } from 'langchain/chat_models/openai';
import { HumanMessage } from 'langchain/schema';
import { z } from 'zod';
import { zodToJsonSchema } from 'zod-to-json-schema';

import './env.js';

async function basic() {
  const extractionFunctionSchema = {
    name: 'extractor',
    description: 'Extracts fields from the input.',
    parameters: {
      type: 'object',
      properties: {
        tone: {
          type: 'string',
          enum: ['positive', 'negative'],
          description: 'The overall tone of the input',
        },
        word_count: {
          type: 'number',
          description: 'The number of words in the input',
        },
        chat_response: {
          type: 'string',
          description: "A response to the human's input",
        },
      },
      required: ['tone', 'word_count', 'chat_response'],
    },
  };
  const model = new ChatOpenAI({
    modelName: 'gpt-4',
    verbose: true,
  }).bind({
    functions: [extractionFunctionSchema],
    function_call: { name: 'extractor' },
  });
  const result = await model.invoke([new HumanMessage('What a beautiful day!')]);
  /*
 AIMessage {
  ...
  additional_kwargs: {
    function_call: {
      name: 'extractor',
      arguments: '{\n' +
      '  "tone": "positive",\n' +
      '  "word_count": 4,\n' +
      '  "chat_response": "Yes, it is indeed a beautiful day!"\n' +
      '}'
    }
  }
}
*/
}

async function usingZod() {
  const extractionFunctionZodSchema = z.object({
    tone: z.enum(['positive', 'negative']).describe('The overall tone of the input'),
    entity: z.string().describe('The entity mentioned in the input'),
    word_count: z.number().describe('The number of words in the input'),
    chat_response: z.string().describe("A response to the human's input"),
    final_punctuation: z.optional(z.string()).describe('The final punctuation mark in the input, if any.'),
  });

  // Bind function arguments to the model.
  // "functions.parameters" must be formatted as JSON Schema.
  // We translate the above Zod schema into JSON schema using the "zodToJsonSchema" package.
  // Omit "function_call" if you want the model to choose a function to call.
  const model = new ChatOpenAI({
    modelName: 'gpt-4',
    verbose: true,
  }).bind({
    functions: [
      {
        name: 'extractor',
        description: 'Extracts fields from the input.',
        parameters: zodToJsonSchema(extractionFunctionZodSchema),
      },
    ],
    function_call: { name: 'extractor' },
  });

  const result = await model.invoke([new HumanMessage('What a beautiful day!')]);

  /*
    AIMessage {
      content: '',
      name: undefined,
      additional_kwargs: {
        function_call: {
          name: 'extractor',
          arguments: '{\n' +
            '  "tone": "positive",\n' +
            '  "entity": "day",\n' +
            '  "word_count": 4,\n' +
            '  "chat_response": "It certainly is a gorgeous day!",\n' +
            '  "final_punctuation": "!"\n' +
            '}'
        }
      }
    }
  */
}
