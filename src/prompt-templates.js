import {
  ChatPromptTemplate,
  HumanMessagePromptTemplate,
  PipelinePromptTemplate,
  PromptTemplate,
  SystemMessagePromptTemplate,
} from 'langchain/prompts';

async function basic() {
  const prompt = PromptTemplate.fromTemplate(
    `You are a naming consultant for new companies.
What is a good name for a company that makes {product}?`
  );

  const formattedPrompt = await prompt.format({
    product: 'colorful socks',
  });

  const noInputPrompt = new PromptTemplate({
    inputVariables: [],
    template: 'Tell me a joke.',
  });
  const formattedNoInputPrompt = await noInputPrompt.format();

  console.log(formattedNoInputPrompt);
  // "Tell me a joke."

  // An example prompt with one input variable
  const oneInputPrompt = new PromptTemplate({
    inputVariables: ['adjective'],
    template: 'Tell me a {adjective} joke.',
  });
  const formattedOneInputPrompt = await oneInputPrompt.format({
    adjective: 'funny',
  });

  console.log(formattedOneInputPrompt);
  // "Tell me a funny joke."

  // An example prompt with multiple input variables
  const multipleInputPrompt = new PromptTemplate({
    inputVariables: ['adjective', 'content'],
    template: 'Tell me a {adjective} joke about {content}.',
  });
  const formattedMultipleInputPrompt = await multipleInputPrompt.format({
    adjective: 'funny',
    content: 'chickens',
  });

  console.log(formattedMultipleInputPrompt);

  const systemMessagePrompt = SystemMessagePromptTemplate.fromTemplate(
    'You are a helpful assistant that translates {input_language} to {output_language}.'
  );
  const humanMessagePrompt = HumanMessagePromptTemplate.fromTemplate('{text}');
  const formatSystemMessagePrompt = await systemMessagePrompt.format({
    input_language: 'English',
    output_language: 'French',
  });
  const formatHumanMessagePrompt = await humanMessagePrompt.format({
    text: 'I love programming.',
  });

  console.log(formatSystemMessagePrompt);

  const chatPrompt = ChatPromptTemplate.fromPromptMessages([systemMessagePrompt, humanMessagePrompt]);

  // Format the messages
  const formattedChatPrompt = await chatPrompt.formatMessages({
    input_language: 'English',
    output_language: 'French',
    text: 'I love programming.',
  });

  console.log(formattedChatPrompt);
}

async function partial() {
  const prompt = new PromptTemplate({
    template: '{foo}{bar}',
    inputVariables: ['foo', 'bar'],
  });

  const partialPrompt = await prompt.partial({
    foo: 'foo',
  });

  const formattedPrompt = await partialPrompt.format({
    bar: 'baz',
  });

  console.log(formattedPrompt);

  function getCurrentDate() {
    return new Date().toISOString();
  }

  const prompt2 = new PromptTemplate({
    template: 'Tell me a {adjective} joke about the day {date}',
    inputVariables: ['adjective', 'date'],
  });

  const partialPrompt2 = await prompt2.partial({
    date: getCurrentDate,
  });

  const formattedPrompt2 = await partialPrompt2.format({
    adjective: 'funny',
  });

  console.log(formattedPrompt2);
}

async function composition() {
  const fullPrompt = PromptTemplate.fromTemplate(`{introduction}

{example}

{start}`);

  const introductionPrompt = PromptTemplate.fromTemplate(`You are impersonating {person}.`);

  const examplePrompt = PromptTemplate.fromTemplate(`Here's an example of an interaction:
Q: {example_q}
A: {example_a}`);

  const startPrompt = PromptTemplate.fromTemplate(`Now, do this for real!
Q: {input}
A:`);

  const composedPrompt = new PipelinePromptTemplate({
    pipelinePrompts: [
      {
        name: 'introduction',
        prompt: introductionPrompt,
      },
      {
        name: 'example',
        prompt: examplePrompt,
      },
      {
        name: 'start',
        prompt: startPrompt,
      },
    ],
    finalPrompt: fullPrompt,
  });

  const formattedPrompt = await composedPrompt.format({
    person: 'Elon Musk',
    example_q: `What's your favorite car?`,
    example_a: 'Telsa',
    input: `What's your favorite social media site?`,
  });

  console.log(formattedPrompt);
}
