import type { Metadata } from 'next';
import { ChatConsole } from '@/components/ChatConsole';

export const metadata: Metadata = {
  title: 'Query — LocalMind',
  description: 'Query the RAG gateway and read the answer with its citations.',
};

export default function QueryPage() {
  return (
    <div className="shell">
      <section className="section" aria-labelledby="q">
        <div className="section__head">
          <h2 id="q">Query</h2>
          <p>
            One request through the agent: route, retrieve, grade, generate, verify. The answer is shown
            with the chunks it cites. An answer without citations is reported as such rather than dressed
            up as a grounded one.
          </p>
        </div>
        <div className="notice" style={{ marginTop: 0, marginBottom: '2rem' }}>
          <div className="notice__hatch" aria-hidden="true" />
          <div className="notice__body">
            <h2>Generation quality is not a LocalMind result</h2>
            <p style={{ marginBottom: 0 }}>
              Answers here are synthesised by Qwen3-4B-Instruct via Ollama. LocalMind-31M is the control
              plane — routing, rewriting, grading — and it is <strong>not trained</strong>. Nothing you read
              in a generated answer is evidence about the 31M model.
            </p>
          </div>
        </div>
        <ChatConsole />
      </section>
    </div>
  );
}
