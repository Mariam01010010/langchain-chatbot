/* RETRIEVAL Chain
Retrieval is useful when you have too much data to pass to the LLM directly. You can then use a retriever to fetch only the most relevant pieces and pass those in.

In this process, we will look up relevant documents from a Retriever and then pass them into the prompt. 
A Retriever can be backed by anything - a SQL table, the internet, etc - but in this instance we will populate a vector store and use that as a retriever. 
*/

// RAG Chain: Part 1
/* downloading a document loader to parse data from a webpage
   npm install cheerio
*/
import { CheerioWebBaseLoader } from "langchain/document_loaders/web/cheerio";

const loader = new CheerioWebBaseLoader(
  "https://docs.smith.langchain.com/user_guide"
);

const docs = await loader.load();

// console.log(docs.length);
// console.log(docs[0].pageContent.length);

// RAG Chain: Part 2
/*  document size is too large to pass into a single model call
    split txt into chunks
*/
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

const splitter = new RecursiveCharacterTextSplitter();

const splitDocs = await splitter.splitDocuments(docs);

// console.log(splitDocs.length);
// console.log(splitDocs[0].pageContent.length);

// RAG Chain: Part 3
/*  need to index loaded documents into a vectorstore
    need the "embedding model" and "vectorstore" components
*/
import { config } from "dotenv";
config();

// can use this embedding model to ingest documents into a vectorstore
import { OpenAIEmbeddings } from "@langchain/openai";

const embeddings = new OpenAIEmbeddings();
// console.log("embeddings", embeddings);

// the LangChain vectorstore class will automatically prepare each raw document using the embeddings model
import { MemoryVectorStore } from "langchain/vectorstores/memory";

const vectorstore = await MemoryVectorStore.fromDocuments(
  splitDocs,
  embeddings
);
// console.log("vectorstore", vectorstore);

// RAG Chain: Part 4
/*  creating a retrieval chain
    chain will:
      - take an incoming question
      - look up relevant documents
      - pass those documents along with the original q into an LLM
      - ask to answer the original question
*/

// set up the chain that takes a q + retrieved docs + generates an answer
import { createStuffDocumentsChain } from "langchain/chains/combine_documents";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { ChatOpenAI } from "@langchain/openai";

const chatModel = new ChatOpenAI({});

const prompt =
  ChatPromptTemplate.fromTemplate(`Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}`);

const documentChain = await createStuffDocumentsChain({
  llm: chatModel,
  prompt,
});

// console.log("documentChain", documentChain);

// below is running this by passing documents in directly
/*
import { Document } from "@langchain/core/documents";

const documentExample = await documentChain.invoke({
  input: "what is LangSmith?",
  context: [
    new Document({
      pageContent:
        "LangSmith is a platform for building production-grade LLM applications.",
    }),
  ],
});

console.log(documentExample);
*/
// but we want docs to first come from the retriever we set up
// bc for a given q, can use retriever to dynamically select the most relevant documents + pass those in
import { createRetrievalChain } from "langchain/chains/retrieval";

const retriever = vectorstore.asRetriever();

const retrievalChain = await createRetrievalChain({
  combineDocsChain: documentChain,
  retriever,
});

// can invoke this chain -> returns an object
const result = await retrievalChain.invoke({
  input: "what is LangSmith?",
});

// response from the LLM is in the answer key
console.log(result.answer);
