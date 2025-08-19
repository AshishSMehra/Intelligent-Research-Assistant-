import { useState, useEffect } from 'react';
import { healthAPI } from '../utils/api';
import { Activity, Server, Database, Users, Settings, AlertTriangle } from 'lucide-react';

const AdminPage = () => {
  const [systemHealth, setSystemHealth] = useState<any>(null);

  useEffect(() => {
    const fetchSystemHealth = async () => {
      try {
        const data = await healthAPI.checkHealth();
        setSystemHealth(data);
      } catch (error) {
        console.error('Failed to fetch system health:', error);
        setSystemHealth({ status: 'error', message: 'Failed to connect' });
      }
    };

    fetchSystemHealth();
    const interval = setInterval(fetchSystemHealth, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const systemStats = [
    {
      name: 'System Status',
      value: systemHealth?.status || 'Unknown',
      icon: Activity,
      color: systemHealth?.status === 'ok' ? 'text-green-600' : 'text-red-600',
    },
    {
      name: 'Active Users',
      value: '12',
      icon: Users,
      color: 'text-blue-600',
    },
    {
      name: 'API Requests',
      value: '1,234',
      icon: Server,
      color: 'text-purple-600',
    },
    {
      name: 'Database Size',
      value: '2.4 GB',
      icon: Database,
      color: 'text-orange-600',
    },
  ];

  const recentLogs = [
    { timestamp: '2024-01-15 10:30:45', level: 'INFO', message: 'User admin logged in successfully' },
    { timestamp: '2024-01-15 10:29:12', level: 'INFO', message: 'Document uploaded: research_paper.pdf' },
    { timestamp: '2024-01-15 10:28:33', level: 'WARN', message: 'High memory usage detected: 85%' },
    { timestamp: '2024-01-15 10:27:18', level: 'INFO', message: 'Agent orchestrator started successfully' },
    { timestamp: '2024-01-15 10:26:45', level: 'ERROR', message: 'Failed to process document: timeout' },
  ];

  const getLevelColor = (level: string) => {
    switch (level) {
      case 'INFO':
        return 'text-blue-600 bg-blue-100';
      case 'WARN':
        return 'text-yellow-600 bg-yellow-100';
      case 'ERROR':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  return (
    <div className="p-6">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-gray-900">System Administration</h1>
        <p className="mt-1 text-sm text-gray-600">
          Monitor system health and manage configurations.
        </p>
      </div>

      {/* System Stats */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
        {systemStats.map((stat) => {
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

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* System Health */}
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">System Health</h2>
          </div>
          <div className="px-6 py-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">API Server</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-gray-900">Online</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Database</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-gray-900">Connected</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">Vector Store</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-green-400 rounded-full"></div>
                <span className="text-sm text-gray-900">Operational</span>
              </div>
            </div>
            
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-600">AI Agents</span>
              <div className="flex items-center space-x-2">
                <div className="w-2 h-2 bg-yellow-400 rounded-full"></div>
                <span className="text-sm text-gray-900">3/4 Active</span>
              </div>
            </div>

            <div className="pt-4">
              <button className="w-full px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700">
                Run System Diagnostics
              </button>
            </div>
          </div>
        </div>

        {/* Quick Actions */}
        <div className="bg-white shadow rounded-lg">
          <div className="px-6 py-4 border-b border-gray-200">
            <h2 className="text-lg font-medium text-gray-900">Quick Actions</h2>
          </div>
          <div className="px-6 py-4 space-y-3">
            <button className="w-full flex items-center justify-between px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-50 rounded-md">
              <div className="flex items-center space-x-3">
                <Settings className="h-5 w-5 text-gray-400" />
                <span>System Configuration</span>
              </div>
            </button>
            
            <button className="w-full flex items-center justify-between px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-50 rounded-md">
              <div className="flex items-center space-x-3">
                <Users className="h-5 w-5 text-gray-400" />
                <span>User Management</span>
              </div>
            </button>
            
            <button className="w-full flex items-center justify-between px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-50 rounded-md">
              <div className="flex items-center space-x-3">
                <Database className="h-5 w-5 text-gray-400" />
                <span>Database Backup</span>
              </div>
            </button>
            
            <button className="w-full flex items-center justify-between px-4 py-3 text-left text-sm text-gray-700 hover:bg-gray-50 rounded-md">
              <div className="flex items-center space-x-3">
                <Activity className="h-5 w-5 text-gray-400" />
                <span>Performance Monitoring</span>
              </div>
            </button>
            
            <button className="w-full flex items-center justify-between px-4 py-3 text-left text-sm text-red-700 hover:bg-red-50 rounded-md">
              <div className="flex items-center space-x-3">
                <AlertTriangle className="h-5 w-5 text-red-400" />
                <span>Emergency Shutdown</span>
              </div>
            </button>
          </div>
        </div>
      </div>

      {/* Recent Logs */}
      <div className="mt-8 bg-white shadow rounded-lg">
        <div className="px-6 py-4 border-b border-gray-200">
          <h2 className="text-lg font-medium text-gray-900">Recent System Logs</h2>
        </div>
        <div className="divide-y divide-gray-200">
          {recentLogs.map((log, index) => (
            <div key={index} className="px-6 py-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-3">
                  <div className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${getLevelColor(log.level)}`}>
                    {log.level}
                  </div>
                  <span className="text-sm text-gray-900">{log.message}</span>
                </div>
                <span className="text-xs text-gray-500">{log.timestamp}</span>
              </div>
            </div>
          ))}
        </div>
        <div className="px-6 py-3 border-t border-gray-200">
          <button className="text-sm text-blue-600 hover:text-blue-800">
            View All Logs
          </button>
        </div>
      </div>
    </div>
  );
};

export default AdminPage; 