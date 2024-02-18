// LLM Chain: Part 1
// importing OPENAI_API_KEY from .env file
import { config } from "dotenv";
config();

// importing and initialising the Language Learning Model
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({});

// asking the LLM a question and printing the response
const response = await chatModel.invoke("what is LangSmith?");

// console.log("response", response);
console.log("content", response["content"]);

// LLM Chain: Part 2
/*  guiding the response with a prompt template
    used to convert raw user input to a better input
    to the LLM
*/
import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"],
]);

// combining these into a simple LLM chain:
const chain = prompt.pipe(chatModel);

// can invoke it and ask the same question and print the response
const promptedResponse = await chain.invoke({
  input: "what is LangSmith?",
});

// console.log({ prompt, chain });
// console.log("promptedResponse", promptedResponse);
console.log("prompted content", promptedResponse.content);

// LLM Chain: Part 3
/*  output of a ChatModel is a message
    so this output parser will convert the chat message
    to a string
*/
import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

const llmChain = prompt.pipe(chatModel).pipe(outputParser);

const stringResponse = await llmChain.invoke({
  input: "what is LangSmith?",
});

console.log("stringResponse", stringResponse);
