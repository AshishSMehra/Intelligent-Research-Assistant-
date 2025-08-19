import { useState, useEffect } from 'react';
import { agentsAPI } from '../utils/api';
import { Bot, Activity, Settings, Pause, Loader2 } from 'lucide-react';

interface Agent {
  id: string;
  name: string;
  type: string;
  status: 'active' | 'inactive' | 'error';
  description: string;
  lastActivity: string;
  tasksCompleted: number;
}

const AgentsPage = () => {
  const [agents, setAgents] = useState<Agent[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [selectedAgent, setSelectedAgent] = useState<Agent | null>(null);

  useEffect(() => {
    const fetchAgents = async () => {
      try {
        const response = await agentsAPI.getAgents();
        setAgents(response.agents || []);
      } catch (error) {
        console.error('Failed to fetch agents:', error);
        // Mock data for demo
        setAgents([
          {
            id: '1',
            name: 'Research Planner',
            type: 'planner',
            status: 'active',
            description: 'Plans and coordinates research tasks',
            lastActivity: '2 minutes ago',
            tasksCompleted: 45,
          },
          {
            id: '2',
            name: 'Document Analyzer',
            type: 'research',
            status: 'active',
            description: 'Analyzes and extracts insights from documents',
            lastActivity: '5 minutes ago',
            tasksCompleted: 128,
          },
          {
            id: '3',
            name: 'Logic Reasoner',
            type: 'reasoner',
            status: 'inactive',
            description: 'Performs logical reasoning and analysis',
            lastActivity: '1 hour ago',
            tasksCompleted: 67,
          },
          {
            id: '4',
            name: 'Task Executor',
            type: 'executor',
            status: 'active',
            description: 'Executes planned tasks and workflows',
            lastActivity: '30 seconds ago',
            tasksCompleted: 89,
          },
        ]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchAgents();
  }, []);

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-100';
      case 'inactive':
        return 'text-gray-600 bg-gray-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'active':
        return <Activity className="h-4 w-4" />;
      case 'inactive':
        return <Pause className="h-4 w-4" />;
      case 'error':
        return <Settings className="h-4 w-4" />;
      default:
        return <Pause className="h-4 w-4" />;
    }
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
        <h1 className="text-2xl font-bold text-gray-900">AI Agents</h1>
        <p className="mt-1 text-sm text-gray-600">
          Manage and monitor your AI agents and their activities.
        </p>
      </div>

      {/* Agent Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 mb-8">
        {agents.map((agent) => (
          <div
            key={agent.id}
            className="bg-white shadow rounded-lg p-6 hover:shadow-md transition-shadow cursor-pointer"
            onClick={() => setSelectedAgent(agent)}
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <Bot className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-900">{agent.name}</h3>
                  <p className="text-sm text-gray-500 capitalize">{agent.type}</p>
                </div>
              </div>
              
              <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(agent.status)}`}>
                {getStatusIcon(agent.status)}
                <span className="capitalize">{agent.status}</span>
              </div>
            </div>

            <p className="text-sm text-gray-600 mb-4">{agent.description}</p>

            <div className="space-y-2">
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Last Activity:</span>
                <span className="text-gray-900">{agent.lastActivity}</span>
              </div>
              <div className="flex justify-between text-sm">
                <span className="text-gray-500">Tasks Completed:</span>
                <span className="text-gray-900">{agent.tasksCompleted}</span>
              </div>
            </div>

            <div className="mt-4 flex space-x-2">
              <button className="flex-1 px-3 py-2 text-xs font-medium text-blue-600 bg-blue-50 rounded-md hover:bg-blue-100">
                Configure
              </button>
              <button className="flex-1 px-3 py-2 text-xs font-medium text-gray-600 bg-gray-50 rounded-md hover:bg-gray-100">
                View Logs
              </button>
            </div>
          </div>
        ))}
      </div>

      {/* Agent Details Modal */}
      {selectedAgent && (
        <div className="fixed inset-0 z-50 overflow-y-auto">
          <div className="flex items-center justify-center min-h-screen pt-4 px-4 pb-20 text-center sm:block sm:p-0">
            <div className="fixed inset-0 transition-opacity" onClick={() => setSelectedAgent(null)}>
              <div className="absolute inset-0 bg-gray-500 opacity-75"></div>
            </div>

            <div className="inline-block align-bottom bg-white rounded-lg px-4 pt-5 pb-4 text-left overflow-hidden shadow-xl transform transition-all sm:my-8 sm:align-middle sm:max-w-lg sm:w-full sm:p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-blue-100 rounded-lg">
                    <Bot className="h-6 w-6 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="text-lg font-medium text-gray-900">{selectedAgent.name}</h3>
                    <p className="text-sm text-gray-500 capitalize">{selectedAgent.type} Agent</p>
                  </div>
                </div>
                
                <div className={`flex items-center space-x-1 px-2 py-1 rounded-full text-xs font-medium ${getStatusColor(selectedAgent.status)}`}>
                  {getStatusIcon(selectedAgent.status)}
                  <span className="capitalize">{selectedAgent.status}</span>
                </div>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700">Description</label>
                  <p className="mt-1 text-sm text-gray-900">{selectedAgent.description}</p>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Last Activity</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedAgent.lastActivity}</p>
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700">Tasks Completed</label>
                    <p className="mt-1 text-sm text-gray-900">{selectedAgent.tasksCompleted}</p>
                  </div>
                </div>

                <div className="flex space-x-3 pt-4">
                  <button className="flex-1 px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-md hover:bg-blue-700">
                    {selectedAgent.status === 'active' ? 'Pause Agent' : 'Start Agent'}
                  </button>
                  <button className="flex-1 px-4 py-2 bg-gray-200 text-gray-900 text-sm font-medium rounded-md hover:bg-gray-300">
                    Configure
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* System Overview */}
      <div className="bg-white shadow rounded-lg p-6">
        <h2 className="text-lg font-medium text-gray-900 mb-4">System Overview</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">
              {agents.filter(a => a.status === 'active').length}
            </div>
            <div className="text-sm text-gray-500">Active Agents</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {agents.reduce((sum, agent) => sum + agent.tasksCompleted, 0)}
            </div>
            <div className="text-sm text-gray-500">Total Tasks Completed</div>
          </div>
          
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">
              {agents.length}
            </div>
            <div className="text-sm text-gray-500">Total Agents</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default AgentsPage; 