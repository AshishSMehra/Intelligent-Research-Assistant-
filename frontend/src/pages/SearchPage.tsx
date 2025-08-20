import { useState } from 'react';
import { searchAPI } from '../utils/api';
import { Search, FileText, Calendar, Loader2 } from 'lucide-react';

interface SearchResult {
  id: string;
  content: string;
  document_id: string;
  score: number;
  metadata?: {
    page_numbers?: number[];
    document_name?: string;
  };
}

const SearchPage: React.FC = () => {
  const [query, setQuery] = useState('');
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim()) return;

    setIsLoading(true);
    setHasSearched(true);

    try {
      const response = await searchAPI.search(query);
      setResults(response.results || []);
    } catch (error) {
      console.error('Search error:', error);
      setResults([]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Search Knowledge Base</h1>
        <p className="mt-1 text-sm text-gray-600">
          Search through your uploaded documents and get relevant information.
        </p>
      </div>

      {/* Search Form */}
      <form onSubmit={handleSearch} className="mb-8">
        <div className="flex space-x-3">
          <div className="flex-1 relative">
            <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
            <input
              type="text"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search your documents..."
              className="w-full pl-10 pr-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
              disabled={isLoading}
            />
          </div>
          <button
            type="submit"
            disabled={isLoading || !query.trim()}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed flex items-center space-x-2"
          >
            {isLoading ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                <span>Searching...</span>
              </>
            ) : (
              <>
                <Search className="h-5 w-5" />
                <span>Search</span>
              </>
            )}
          </button>
        </div>
      </form>

      {/* Results */}
      {hasSearched && (
        <div className="space-y-6">
          {isLoading ? (
            <div className="text-center py-12">
              <Loader2 className="h-8 w-8 animate-spin mx-auto text-blue-600 mb-4" />
              <p className="text-gray-600">Searching your documents...</p>
            </div>
          ) : results.length > 0 ? (
            <div>
              <h2 className="text-lg font-medium text-gray-900 mb-4">
                Found {results.length} result{results.length !== 1 ? 's' : ''}
              </h2>
              <div className="space-y-4">
                {results.map((result, index) => (
                  <div key={result.id || index} className="bg-white shadow rounded-lg p-6">
                    <div className="flex items-start justify-between mb-2">
                      <div className="flex items-center space-x-2">
                        <FileText className="h-5 w-5 text-gray-400" />
                        <span className="text-sm font-medium text-gray-900">
                          {result.metadata?.document_name || `Document ${result.document_id}`}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2 text-sm text-gray-500">
                        <Calendar className="h-4 w-4" />
                        <span>Score: {(result.score * 100).toFixed(1)}%</span>
                      </div>
                    </div>
                    <p className="text-gray-700 leading-relaxed">{result.content}</p>
                    {result.metadata?.page_numbers && (
                      <p className="text-sm text-gray-500 mt-2">
                        Page{result.metadata.page_numbers.length > 1 ? 's' : ''}: {result.metadata.page_numbers.join(', ')}
                      </p>
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div className="text-center py-12">
              <Search className="h-12 w-12 text-gray-400 mx-auto mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">No results found</h3>
              <p className="text-gray-600">
                Try adjusting your search terms or upload more documents to search through.
              </p>
            </div>
          )}
        </div>
      )}

      {/* Search Tips */}
      {!hasSearched && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-6">
          <h3 className="text-sm font-medium text-blue-900 mb-2">Search Tips</h3>
          <ul className="text-sm text-blue-800 space-y-1">
            <li>• Use specific keywords for better results</li>
            <li>• Try different variations of your search terms</li>
            <li>• Search is performed across all uploaded documents</li>
            <li>• Results are ranked by relevance to your query</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default SearchPage; 