import { useState, useCallback } from 'react';
import { uploadAPI } from '../utils/api';
import { Upload, File, CheckCircle, XCircle, Loader2 } from 'lucide-react';

interface UploadedFile {
  name: string;
  size: number;
  status: 'uploading' | 'success' | 'error';
  message?: string;
}

const UploadPage = () => {
  const [files, setFiles] = useState<UploadedFile[]>([]);
  const [isDragOver, setIsDragOver] = useState(false);

  const handleFileUpload = async (file: File) => {
    const uploadedFile: UploadedFile = {
      name: file.name,
      size: file.size,
      status: 'uploading',
    };

    setFiles(prev => [...prev, uploadedFile]);

    try {
      const response = await uploadAPI.uploadFile(file);
      setFiles(prev => prev.map(f => 
        f.name === file.name 
          ? { ...f, status: 'success', message: response.message }
          : f
      ));
    } catch (error) {
      setFiles(prev => prev.map(f => 
        f.name === file.name 
          ? { ...f, status: 'error', message: 'Upload failed' }
          : f
      ));
    }
  };

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    
    const droppedFiles = Array.from(e.dataTransfer.files);
    droppedFiles.forEach(handleFileUpload);
  }, []);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
  }, []);

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFiles = Array.from(e.target.files || []);
    selectedFiles.forEach(handleFileUpload);
  };

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">Upload Documents</h1>
        <p className="mt-1 text-sm text-gray-600">
          Upload your documents for analysis and processing by the AI assistant.
        </p>
      </div>

      {/* Upload Area */}
      <div
        className={`border-2 border-dashed rounded-lg p-8 text-center mb-8 transition-colors ${
          isDragOver
            ? 'border-blue-400 bg-blue-50'
            : 'border-gray-300 hover:border-gray-400'
        }`}
        onDrop={handleDrop}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
      >
        <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 mb-2">
          Drop files here or click to upload
        </h3>
        <p className="text-sm text-gray-600 mb-4">
          Supports PDF, DOC, DOCX, TXT files up to 10MB
        </p>
        <label className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 cursor-pointer">
          Choose Files
          <input
            type="file"
            multiple
            accept=".pdf,.doc,.docx,.txt"
            onChange={handleFileSelect}
            className="hidden"
          />
        </label>
      </div>

      {/* Upload Progress */}
      {files.length > 0 && (
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Upload Progress</h2>
          </div>
          <div className="divide-y divide-gray-200">
            {files.map((file, index) => (
              <div key={index} className="px-6 py-4 flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <File className="h-8 w-8 text-gray-400" />
                  <div>
                    <p className="text-sm font-medium text-gray-900">{file.name}</p>
                    <p className="text-xs text-gray-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-3">
                  {file.status === 'uploading' && (
                    <div className="flex items-center space-x-2 text-blue-600">
                      <Loader2 className="h-4 w-4 animate-spin" />
                      <span className="text-sm">Uploading...</span>
                    </div>
                  )}
                  
                  {file.status === 'success' && (
                    <div className="flex items-center space-x-2 text-green-600">
                      <CheckCircle className="h-5 w-5" />
                      <span className="text-sm">Success</span>
                    </div>
                  )}
                  
                  {file.status === 'error' && (
                    <div className="flex items-center space-x-2 text-red-600">
                      <XCircle className="h-5 w-5" />
                      <span className="text-sm">Failed</span>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Upload Tips */}
      <div className="mt-8 bg-blue-50 border border-blue-200 rounded-lg p-6">
        <h3 className="text-sm font-medium text-blue-900 mb-2">Upload Tips</h3>
        <ul className="text-sm text-blue-800 space-y-1">
          <li>• Supported formats: PDF, DOC, DOCX, TXT</li>
          <li>• Maximum file size: 10MB per file</li>
          <li>• Files are automatically processed and indexed for search</li>
          <li>• Processing may take a few minutes for large documents</li>
        </ul>
      </div>
    </div>
  );
};

export default UploadPage; 