"use client"

import Image from "next/image";

import { useEffect } from 'react';



let tf2:any;
if (process.env.NEXT_PUBLIC_IS_SERVER === 'true') {
  tf2 = require('@tensorflow/tfjs-node');
} else {
  tf2 = require('@tensorflow/tfjs');
}
 // Display forecasted savings


export default function Home() {
  let tf:any;
  useEffect(() => {
    async function loadTensorFlow() {
      
      if (typeof window === 'undefined') {
        // Server-side: Load @tensorflow/tfjs-node
        tf = await import('@tensorflow/tfjs-node');
      } else {
        // Client-side: Load @tensorflow/tfjs
        tf = await import('@tensorflow/tfjs');
      }

      // Use TensorFlow.js here
    }

    loadTensorFlow();
  }, []);

// Example data (replace these with your actual data)
const savingsData = [10220, 12125, 52400, 98797, 114166]; 
const expenditureData = [1506, 6581, 9455, 10814, 13777]; 

// Convert the data to a tensor
const savingsTensor = tf.tensor(savingsData, [savingsData.length, 1]);
const expenditureTensor = tf.tensor(expenditureData, [expenditureData.length, 1]);

// Calculate the mean
const savingsMean = savingsTensor.mean();
const expenditureMean = expenditureTensor.mean();

// Calculate the standard deviation
const savingsStd = tf.sqrt(savingsTensor.sub(savingsMean).pow(2).mean());
const expenditureStd = tf.sqrt(expenditureTensor.sub(expenditureMean).pow(2).mean());

// Normalize the data
const savingsNormalized = savingsTensor.sub(savingsMean).div(savingsStd);
const expenditureNormalized = expenditureTensor.sub(expenditureMean).div(expenditureStd);

// Create sequences (windows) of data for the LSTM model
const WINDOW_SIZE = 12; // Use 12 months of data to predict the next value

function createSequences(data:any, windowSize:any) {
  let inputs = [];
  let labels = [];
  for (let i = 0; i < data.length - windowSize; i++) {
    let inputSequence = data.slice(i, i + windowSize);
    let label = data[i + windowSize];
    inputs.push(inputSequence);
    labels.push(label);
  }
  return [tf.tensor2d(inputs), tf.tensor2d(labels)];
}

const [X, y] = createSequences(expenditureNormalized.arraySync(), WINDOW_SIZE);

// Build the LSTM model
const model = tf.sequential();
model.add(tf.layers.lstm({ units: 50, returnSequences: false, inputShape: [WINDOW_SIZE, 1] }));
model.add(tf.layers.dense({ units: 1 }));

model.compile({
  optimizer: 'adam',
  loss: 'meanSquaredError',
});

// Train the model
// @ts-expect-error

await model.fit(X, y, {
  epochs: 20,
  batchSize: 16,
});

// Forecast the next 12 months of savings based on current account expenditure
async function forecastFutureSavings(model:any, recentExpenditure:any) {
  let futureSavings = [];
  let input = recentExpenditure.slice(-WINDOW_SIZE);

  for (let i = 0; i < 12; i++) {
    const prediction = model.predict(tf.tensor2d([input]));
    futureSavings.push(prediction.arraySync()[0][0]);
    input = input.slice(1).concat(prediction.arraySync()[0][0]);
  }
  
  return futureSavings;
}
// @ts-expect-error

const forecastedSavings:any = await forecastFutureSavings(model, expenditureNormalized.arraySync());

// Denormalize the forecasted savings
// @ts-expect-error
const denormalizedForecast = forecastedSavings.map(s => s * savingsStd.arraySync() + savingsMean.arraySync() );

console.log(denormalizedForecast);
  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="z-10 w-full max-w-5xl items-center justify-between font-mono text-sm lg:flex">
        <p className="fixed left-0 top-0 flex w-full justify-center border-b border-gray-300 bg-gradient-to-b from-zinc-200 pb-6 pt-8 backdrop-blur-2xl dark:border-neutral-800 dark:bg-zinc-800/30 dark:from-inherit lg:static lg:w-auto  lg:rounded-xl lg:border lg:bg-gray-200 lg:p-4 lg:dark:bg-zinc-800/30">
          Get started by editing&nbsp;
          <code className="font-mono font-bold">app/page.tsx</code>
        </p>
        <div className="fixed bottom-0 left-0 flex h-48 w-full items-end justify-center bg-gradient-to-t from-white via-white dark:from-black dark:via-black lg:static lg:size-auto lg:bg-none">
          <a
            className="pointer-events-none flex place-items-center gap-2 p-8 lg:pointer-events-auto lg:p-0"
            href="https://vercel.com?utm_source=create-next-app&utm_medium=appdir-template&utm_campaign=create-next-app"
            target="_blank"
            rel="noopener noreferrer"
          >
            By{" "}
            <Image
              src="/vercel.svg"
              alt="Vercel Logo"
              className="dark:invert"
              width={100}
              height={24}
              priority
            />
          </a>
        </div>
      </div>

      <div className="relative z-[-1] flex place-items-center before:absolute before:h-[300px] before:w-full before:-translate-x-1/2 before:rounded-full before:bg-gradient-radial before:from-white before:to-transparent before:blur-2xl before:content-[''] after:absolute after:-z-20 after:h-[180px] after:w-full after:translate-x-1/3 after:bg-gradient-conic after:from-sky-200 after:via-blue-200 after:blur-2xl after:content-[''] before:dark:bg-gradient-to-br before:dark:from-transparent before:dark:to-blue-700 before:dark:opacity-10 after:dark:from-sky-900 after:dark:via-[#0141ff] after:dark:opacity-40 sm:before:w-[480px] sm:after:w-[240px] before:lg:h-[360px]">
        <Image
          className="relative dark:drop-shadow-[0_0_0.3rem_#ffffff70] dark:invert"
          src="/next.svg"
          alt="Next.js Logo"
          width={180}
          height={37}
          priority
        />
      </div>

      <div className="mb-32 grid text-center lg:mb-0 lg:w-full lg:max-w-5xl lg:grid-cols-4 lg:text-left">
        <a
          href="https://nextjs.org/docs?utm_source=create-next-app&utm_medium=appdir-template&utm_campaign=create-next-app"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-2xl font-semibold">
            Docs{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Find in-depth information about Next.js features and API.
          </p>
        </a>

        <a
          href="https://nextjs.org/learn?utm_source=create-next-app&utm_medium=appdir-template-tw&utm_campaign=create-next-app"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-2xl font-semibold">
            Learn{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Learn about Next.js in an interactive course with&nbsp;quizzes!
          </p>
        </a>

        <a
          href="https://vercel.com/templates?framework=next.js&utm_source=create-next-app&utm_medium=appdir-template&utm_campaign=create-next-app"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-2xl font-semibold">
            Templates{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-sm opacity-50">
            Explore starter templates for Next.js.
          </p>
        </a>

        <a
          href="https://vercel.com/new?utm_source=create-next-app&utm_medium=appdir-template&utm_campaign=create-next-app"
          className="group rounded-lg border border-transparent px-5 py-4 transition-colors hover:border-gray-300 hover:bg-gray-100 hover:dark:border-neutral-700 hover:dark:bg-neutral-800/30"
          target="_blank"
          rel="noopener noreferrer"
        >
          <h2 className="mb-3 text-2xl font-semibold">
            Deploy{" "}
            <span className="inline-block transition-transform group-hover:translate-x-1 motion-reduce:transform-none">
              -&gt;
            </span>
          </h2>
          <p className="m-0 max-w-[30ch] text-balance text-sm opacity-50">
            Instantly deploy your Next.js site to a shareable URL with Vercel.
          </p>
        </a>
      </div>
    </main>
  );
}
