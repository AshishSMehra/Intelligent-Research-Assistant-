import { useState, useEffect } from 'react';
import { documentsAPI } from '../utils/api';
import { FileText, Download, Trash2, Eye, Calendar, Loader2 } from 'lucide-react';

interface Document {
  id: string;
  filename: string;
  size: number;
  type: string;
  uploaded_at: string;
  status: string;
  pages?: number;
  processed?: boolean;
}

const DocumentsPage = () => {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null);

  useEffect(() => {
    const fetchDocuments = async () => {
      try {
        const response = await documentsAPI.getDocuments();
        setDocuments(response.documents || []);
      } catch (error) {
        console.error('Failed to fetch documents:', error);
        // Mock data for demo
        setDocuments([
          {
            id: '1',
            filename: 'Research Paper - AI in Healthcare.pdf',
            size: 2456789,
            type: 'pdf',
            uploaded_at: '2024-01-15T10:30:00Z',
            status: 'processed',
            pages: 24,
            processed: true,
          },
          {
            id: '2',
            filename: 'Market Analysis Report.docx',
            size: 1234567,
            type: 'docx',
            uploaded_at: '2024-01-14T15:45:00Z',
            status: 'processed',
            pages: 12,
            processed: true,
          },
          {
            id: '3',
            filename: 'Technical Specifications.txt',
            size: 45678,
            type: 'txt',
            uploaded_at: '2024-01-14T09:20:00Z',
            status: 'processing',
            processed: false,
          },
          {
            id: '4',
            filename: 'Financial Data Q4.pdf',
            size: 3456789,
            type: 'pdf',
            uploaded_at: '2024-01-13T14:15:00Z',
            status: 'processed',
            pages: 45,
            processed: true,
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchDocuments();
  }, []);

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleDateString('en-US', {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'processed':
        return 'text-green-600 bg-green-100';
      case 'processing':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getFileIcon = () => {
    return <FileText className="h-6 w-6 text-blue-600" />;
  };

  if (isLoading) {
    return (
      <div className="p-6">
        <div className="flex items-center justify-center h-64">
          <Loader2 className="h-8 w-8 animate-spin text-blue-600" />
        </div>
      </div>
    );
  }

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Documents</h1>
        <p className="mt-1 text-sm text-gray-600">
          View and manage your uploaded documents.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6 mb-8">
        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <FileText className="h-6 w-6 text-blue-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Total Documents
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {documents.length}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Eye className="h-6 w-6 text-green-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Processed
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {documents.filter(d => d.status === 'processed').length}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Loader2 className="h-6 w-6 text-yellow-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Processing
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {documents.filter(d => d.status === 'processing').length}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>

        <div className="bg-white overflow-hidden shadow rounded-lg">
          <div className="p-5">
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <Calendar className="h-6 w-6 text-purple-600" />
              </div>
              <div className="ml-5 w-0 flex-1">
                <dl>
                  <dt className="text-sm font-medium text-gray-500 truncate">
                    Total Size
                  </dt>
                  <dd className="text-lg font-medium text-gray-900">
                    {formatFileSize(documents.reduce((sum, doc) => sum + doc.size, 0))}
                  </dd>
                </dl>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Documents List */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">All Documents</h2>
        </div>

        {documents.length > 0 ? (
          <div className="divide-y divide-gray-200">
            {documents.map((document) => (
              <div key={document.id} className="px-6 py-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-4">
                    {getFileIcon()}
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-gray-900 truncate">
                        {document.filename}
                      </p>
                      <div className="flex items-center space-x-4 mt-1">
                        <span className="text-xs text-gray-500">
                          {formatFileSize(document.size)}
                        </span>
                        {document.pages && (
                          <span className="text-xs text-gray-500">
                            {document.pages} pages
                          </span>
                        )}
                        <span className="text-xs text-gray-500">
                          {formatDate(document.uploaded_at)}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="flex items-center space-x-4">
                    <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(document.status)}`}>
                      {document.status === 'processing' && (
                        <Loader2 className="h-3 w-3 animate-spin mr-1" />
                      )}
                      {document.status}
                    </div>

                    <div className="flex items-center space-x-2">
                      <button
                        onClick={() => setSelectedDocument(document)}
                        className="text-blue-600 hover:text-blue-800 p-1"
                        title="View Details"
                      >
                        <Eye className="h-4 w-4" />
                      </button>
                      <button
                        className="text-green-600 hover:text-green-800 p-1"
                        title="Download"
                      >
                        <Download className="h-4 w-4" />
                      </button>
                      <button
                        className="text-red-600 hover:text-red-800 p-1"
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="px-6 py-8 text-center">
            <FileText className="h-12 w-12 text-gray-400 mx-auto mb-4" />
            <h3 className="text-lg font-medium text-gray-900 mb-2">No documents found</h3>
            <p className="text-gray-600">
              Upload some documents to get started.
            </p>
          </div>
        )}
      </div>

      {/* Document Details Modal */}
      {selectedDocument && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" onClick={() => setSelectedDocument(null)}>
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
              <div className="flex items-center space-x-3 mb-4">
                {getFileIcon()}
                <div>
                  <h3 className="text-lg font-medium text-gray-900">{selectedDocument.filename}</h3>
                  <p className="text-sm text-gray-500">{selectedDocument.type.toUpperCase()} Document</p>
                </div>
              </div>

              <div className="space-y-4">
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">File Size</label>
                    <p className="mt-1 text-sm text-gray-900">{formatFileSize(selectedDocument.size)}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Status</label>
                    <div className={`mt-1 inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getStatusColor(selectedDocument.status)}`}>
                      {selectedDocument.status}
                    </div>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700">Uploaded</label>
                  <p className="mt-1 text-sm text-gray-900">{formatDate(selectedDocument.uploaded_at)}</p>
                </div>

                {selectedDocument.pages && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Pages</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedDocument.pages}</p>
                  </div>
                )}

                <div className="flex space-x-3 pt-4">
                  <button className="flex-1 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700">
                    Download
                  </button>
                  <button className="flex-1 px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-md hover:bg-red-700">
                    Delete
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default DocumentsPage; 