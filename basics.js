import { config } from "dotenv";
config();

import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({});

const response = await chatModel.invoke("what is LangSmith?");

console.log(response);
