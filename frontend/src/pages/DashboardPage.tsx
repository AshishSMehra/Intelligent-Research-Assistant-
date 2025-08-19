import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { healthAPI } from '../utils/api';
import { 
  MessageSquare, 
  Upload, 
  Search, 
  Bot, 
  FileText, 
  Activity,
  Users,
  Database
} from 'lucide-react';

const DashboardPage = () => {
  const { user } = useAuth();
  const [healthStatus, setHealthStatus] = useState<any>(null);

  useEffect(() => {
    const fetchHealth = async () => {
      try {
        const data = await healthAPI.checkHealth();
        setHealthStatus(data);
      } catch (error) {
        console.error('Failed to fetch health status:', error);
      }
    };

    fetchHealth();
  }, []);

  const quickActions = [
    {
      name: 'Start Chat',
      description: 'Ask questions and get AI-powered answers',
      href: '/chat',
      icon: MessageSquare,
      color: 'bg-blue-500',
    },
    {
      name: 'Upload Document',
      description: 'Upload files for analysis and processing',
      href: '/upload',
      icon: Upload,
      color: 'bg-green-500',
    },
    {
      name: 'Search Knowledge',
      description: 'Search through your document collection',
      href: '/search',
      icon: Search,
      color: 'bg-purple-500',
    },
    {
      name: 'Manage Agents',
      description: 'Configure and monitor AI agents',
      href: '/agents',
      icon: Bot,
      color: 'bg-orange-500',
    },
  ];

  const stats = [
    {
      name: 'Active Sessions',
      value: '12',
      icon: Users,
      color: 'text-blue-600',
    },
    {
      name: 'Documents Processed',
      value: '1,234',
      icon: FileText,
      color: 'text-green-600',
    },
    {
      name: 'Queries Today',
      value: '89',
      icon: Database,
      color: 'text-purple-600',
    },
    {
      name: 'System Status',
      value: healthStatus?.status || 'Unknown',
      icon: Activity,
      color: healthStatus?.status === 'ok' ? 'text-green-600' : 'text-red-600',
    },
  ];

  return (
    <div className="p-6">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">
          Welcome back, {user?.username}!
        </h1>
        <p className="mt-1 text-sm text-gray-600">
          Here's what's happening with your research assistant today.
        </p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {stats.map((stat) => {
          const Icon = stat.icon;
          return (
            <div key={stat.name} className="bg-white overflow-hidden shadow rounded-lg">
              <div className="p-5">
                <div className="flex items-center">
                  <div className="flex-shrink-0">
                    <Icon className={`h-6 w-6 ${stat.color}`} />
                  </div>
                  <div className="ml-5 w-0 flex-1">
                    <dl>
                      <dt className="text-sm font-medium text-gray-500 truncate">
                        {stat.name}
                      </dt>
                      <dd className="text-lg font-medium text-gray-900">
                        {stat.value}
                      </dd>
                    </dl>
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Quick Actions */}
      <div className="mb-8">
        <h2 className="text-lg font-medium text-gray-900 mb-4">Quick Actions</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {quickActions.map((action) => {
            const Icon = action.icon;
            return (
              <a
                key={action.name}
                href={action.href}
                className="bg-white overflow-hidden shadow rounded-lg hover:shadow-md transition-shadow"
              >
                <div className="p-6">
                  <div className="flex items-center">
                    <div className={`flex-shrink-0 ${action.color} rounded-md p-3`}>
                      <Icon className="h-6 w-6 text-white" />
                    </div>
                    <div className="ml-4">
                      <h3 className="text-sm font-medium text-gray-900">
                        {action.name}
                      </h3>
                      <p className="text-sm text-gray-500 mt-1">
                        {action.description}
                      </p>
                    </div>
                  </div>
                </div>
              </a>
            );
          })}
        </div>
      </div>

      {/* Recent Activity */}
      <div className="bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Recent Activity</h2>
        </div>
        <div className="px-6 py-4">
          <div className="space-y-4">
            {[
              { action: 'Document uploaded', time: '2 minutes ago', icon: Upload },
              { action: 'Chat session started', time: '15 minutes ago', icon: MessageSquare },
              { action: 'Search query executed', time: '1 hour ago', icon: Search },
              { action: 'Agent configuration updated', time: '2 hours ago', icon: Bot },
            ].map((item, index) => {
              const Icon = item.icon;
              return (
                <div key={index} className="flex items-center space-x-3">
                  <div className="flex-shrink-0">
                    <Icon className="h-5 w-5 text-gray-400" />
                  </div>
                  <div className="flex-1">
                    <p className="text-sm text-gray-900">{item.action}</p>
                    <p className="text-xs text-gray-500">{item.time}</p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
};

export default DashboardPage; 