// LLM Chain: Part 1
import { config } from "dotenv";
config();

import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({});

const response = await chatModel.invoke("what is LangSmith?");

// console.log("response", response);
console.log("content", response["content"]);

// import { Configuration, OpenAIApi } from "openai";
// import { ChatOpenAI } from "@langchain/openai";

// LLM Chain: Part 2
import { ChatPromptTemplate } from "@langchain/core/prompts";

const prompt = ChatPromptTemplate.fromMessages([
  ["system", "You are a world class technical documentation writer."],
  ["user", "{input}"],
]);

const chain = prompt.pipe(chatModel);
const promptedResponse = await chain.invoke({
  input: "what is LangSmith?",
});

// console.log({ prompt, chain });
// console.log("promptedResponse", promptedResponse);
console.log("prompted content", promptedResponse.content);

// LLM Chain: Part 3
import { StringOutputParser } from "@langchain/core/output_parsers";

const outputParser = new StringOutputParser();

const llmChain = prompt.pipe(chatModel).pipe(outputParser);

const stringResponse = await llmChain.invoke({
  input: "what is LangSmith?",
});

console.log("stringResponse", stringResponse);
