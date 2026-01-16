export default function Home() {
  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-24">
      <div className="z-10 max-w-5xl w-full items-center justify-center font-mono text-sm">
        <h1 className="text-4xl font-bold text-center mb-4">
          AI Knowledge Copilot
        </h1>
        <p className="text-center text-lg mb-8">
          Production-grade RAG system for querying documents and web content
        </p>
        <div className="text-center">
          <p className="text-sm text-gray-600">
            Frontend is ready. Backend API: {process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000'}
          </p>
        </div>
      </div>
    </main>
  )
}
