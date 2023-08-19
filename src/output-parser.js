import { OpenAI } from 'langchain/llms/openai';
import { StructuredOutputParser } from 'langchain/output_parsers';
import { PromptTemplate } from 'langchain/prompts';
import { z } from 'zod';

import './env.js';

async function basic() {
  // With a `StructuredOutputParser` we can define a schema for the output.
  const parser = StructuredOutputParser.fromNamesAndDescriptions({
    answer: "answer to the user's question",
    source: "source used to answer the user's question, should be a website.",
  });

  const formatInstructions = parser.getFormatInstructions();

  const prompt = new PromptTemplate({
    template: 'Answer the users question as best as possible.\n{format_instructions}\n{question}',
    inputVariables: ['question'],
    partialVariables: { format_instructions: formatInstructions },
  });

  const model = new OpenAI({ temperature: 0, verbose: true });

  const input = await prompt.format({
    question: 'What is the capital of France?',
  });
  const response = await model.call(input);

  console.log('input', input);
  /*
  Answer the users question as best as possible.
  You must format your output as a JSON value that adheres to a given "JSON Schema" instance.

  "JSON Schema" is a declarative language that allows you to annotate and validate JSON documents.

  For example, the example "JSON Schema" instance {{"properties": {{"foo": {{"description": "a list of test words", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
  would match an object with one required property, "foo". The "type" property specifies "foo" must be an "array", and the "description" property semantically describes it as "a list of test words". The items within "foo" must be strings.
  Thus, the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of this example "JSON Schema". The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

  Your output will be parsed and type-checked according to the provided schema instance, so make sure all fields in your output match the schema exactly and there are no trailing commas!

  Here is the JSON Schema instance your output must adhere to. Include the enclosing markdown codeblock:
  ```json
  {"type":"object","properties":{"answer":{"type":"string","description":"answer to the user's question"},"source":{"type":"string","description":"source used to answer the user's question, should be a website."}},"required":["answer","source"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#"}
  ```

  What is the capital of France?
  */

  console.log('response', response);
  /*
  {"answer": "Paris", "source": "https://en.wikipedia.org/wiki/Paris"}
  */

  const parsedResponse = await parser.parse(response);
  console.log('parsedResponse', parsedResponse);
  // { answer: 'Paris', source: 'https://en.wikipedia.org/wiki/Paris' }
}

async function usingZod() {
  // We can use zod to define a schema for the output using the `fromZodSchema` method of `StructuredOutputParser`.
  const parser = StructuredOutputParser.fromZodSchema(
    z.object({
      answer: z.string().describe("answer to the user's question"),
      sources: z.array(z.string()).describe('sources used to answer the question, should be websites.'),
    }),
  );

  const formatInstructions = parser.getFormatInstructions();

  const prompt = new PromptTemplate({
    template: 'Answer the users question as best as possible.\n{format_instructions}\n{question}',
    inputVariables: ['question'],
    partialVariables: { format_instructions: formatInstructions },
  });

  const model = new OpenAI({ temperature: 0, verbose: true });

  const input = await prompt.format({
    question: 'What is the capital of France?',
  });
  const response = await model.call(input);

  console.log(input);
  /*
  Answer the users question as best as possible.
  The output should be formatted as a JSON instance that conforms to the JSON schema below.

  As an example, for the schema {{"properties": {{"foo": {{"title": "Foo", "description": "a list of strings", "type": "array", "items": {{"type": "string"}}}}}}, "required": ["foo"]}}}}
  the object {{"foo": ["bar", "baz"]}} is a well-formatted instance of the schema. The object {{"properties": {{"foo": ["bar", "baz"]}}}} is not well-formatted.

  Here is the output schema:
  ```
  {"type":"object","properties":{"answer":{"type":"string","description":"answer to the user's question"},"sources":{"type":"array","items":{"type":"string"},"description":"sources used to answer the question, should be websites."}},"required":["answer","sources"],"additionalProperties":false,"$schema":"http://json-schema.org/draft-07/schema#"}
  ```

  What is the capital of France?
  */

  console.log(response);
  /*
  {"answer": "Paris", "sources": ["https://en.wikipedia.org/wiki/Paris"]}
  */

  console.log(await parser.parse(response));
  /*
  { answer: 'Paris', sources: [ 'https://en.wikipedia.org/wiki/Paris' ] }
  */
}
