import React, { useState, useEffect } from 'react';
import { useDropzone } from 'react-dropzone';
import { leakAPI } from '../services/api';
import { toast } from 'react-toastify';
import { FiAlertTriangle, FiUpload, FiCheck, FiX, FiEye } from 'react-icons/fi';

const LeakDetection = () => {
  const [file, setFile] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [result, setResult] = useState(null);
  const [reports, setReports] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchLeakReports();
  }, []);

  const fetchLeakReports = async () => {
    try {
      const response = await leakAPI.list();
      setReports(response.data.leak_reports);
    } catch (error) {
      toast.error('Ошибка загрузки отчётов');
    } finally {
      setLoading(false);
    }
  };

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    accept: {
      'video/*': ['.mp4', '.avi', '.mov', '.mkv', '.webm']
    },
    maxFiles: 1,
    onDrop: (acceptedFiles) => {
      if (acceptedFiles.length > 0) {
        setFile(acceptedFiles[0]);
        setResult(null);
      }
    },
  });

  const handleAnalyze = async () => {
    if (!file) {
      toast.error('Выберите видеофайл');
      return;
    }

    setAnalyzing(true);
    setResult(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await leakAPI.detect(formData);
      setResult(response.data);
      toast.success('Утечка обнаружена!');
      fetchLeakReports();
    } catch (error) {
      toast.error(error.response?.data?.error || 'Ошибка анализа');
      setResult({ error: error.response?.data?.error || 'Ошибка анализа' });
    } finally {
      setAnalyzing(false);
    }
  };

  const handleUpdateReport = async (reportId, status, banUser = false) => {
    try {
      await leakAPI.update(reportId, { status, ban_user: banUser });
      toast.success('Отчёт обновлён');
      fetchLeakReports();
    } catch (error) {
      toast.error('Ошибка обновления отчёта');
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'investigating': return 'bg-blue-100 text-blue-800';
      case 'confirmed': return 'bg-red-100 text-red-800';
      case 'false_positive': return 'bg-green-100 text-green-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusLabel = (status) => {
    switch (status) {
      case 'pending': return 'Ожидает';
      case 'investigating': return 'Расследуется';
      case 'confirmed': return 'Подтверждено';
      case 'false_positive': return 'Ложное срабатывание';
      default: return status;
    }
  };

  return (
    <div className="p-6 max-w-7xl mx-auto">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-900 flex items-center space-x-3">
          <FiAlertTriangle className="text-red-600" />
          <span>Детекция утечек</span>
        </h1>
        <p className="text-gray-600 mt-2">
          Загрузите подозрительное видео для определения источника утечки
        </p>
      </div>

      {/* Upload and Analysis Section */}
      <div className="bg-white rounded-xl shadow-lg p-6 mb-8">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Анализ видео</h2>
        
        {!file ? (
          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-lg p-12 text-center cursor-pointer transition-colors ${
              isDragActive
                ? 'border-red-500 bg-red-50'
                : 'border-gray-300 hover:border-red-400 hover:bg-gray-50'
            }`}
          >
            <input {...getInputProps()} />
            <FiUpload className="w-16 h-16 mx-auto text-gray-400 mb-4" />
            <p className="text-lg font-semibold text-gray-700 mb-2">
              {isDragActive ? 'Отпустите файл здесь' : 'Загрузите подозрительное видео'}
            </p>
            <p className="text-sm text-gray-500">
              Перетащите или нажмите для выбора
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            <div className="border-2 border-red-300 bg-red-50 rounded-lg p-6">
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-semibold text-gray-900">{file.name}</p>
                  <p className="text-sm text-gray-600">
                    {(file.size / (1024 * 1024)).toFixed(2)} МБ
                  </p>
                </div>
                <button
                  onClick={() => setFile(null)}
                  className="p-2 text-red-500 hover:bg-red-100 rounded-lg"
                >
                  <FiX className="w-6 h-6" />
                </button>
              </div>
            </div>

            <button
              onClick={handleAnalyze}
              disabled={analyzing}
              className="w-full py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 font-semibold transition-colors"
            >
              {analyzing ? 'Анализ...' : 'Анализировать водяной знак'}
            </button>
          </div>
        )}

        {/* Analysis Result */}
        {result && (
          <div className="mt-6">
            {result.error ? (
              <div className="bg-red-50 border-2 border-red-200 rounded-lg p-6">
                <div className="flex items-center space-x-3 mb-2">
                  <FiX className="w-6 h-6 text-red-600" />
                  <h3 className="text-lg font-bold text-red-900">Анализ не удался</h3>
                </div>
                <p className="text-red-700">{result.error}</p>
              </div>
            ) : (
              <div className="bg-green-50 border-2 border-green-200 rounded-lg p-6">
                <div className="flex items-center space-x-3 mb-4">
                  <FiCheck className="w-6 h-6 text-green-600" />
                  <h3 className="text-lg font-bold text-green-900">Утечка обнаружена!</h3>
                </div>
                
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-4">
                  <div>
                    <p className="text-sm text-gray-600">Подозреваемый:</p>
                    <p className="font-bold text-gray-900">{result.suspected_user.full_name}</p>
                    <p className="text-sm text-gray-500">{result.suspected_user.email}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Видео:</p>
                    <p className="font-bold text-gray-900">{result.video.title}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">ID водяного знака:</p>
                    <p className="font-mono font-bold text-gray-900">{result.leak_report.watermark_id}</p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Метод обнаружения:</p>
                    <p className="font-bold text-gray-900 capitalize">
                      {result.leak_report.detection_method === 'manual' ? 'Ручной' : result.leak_report.detection_method}
                    </p>
                  </div>
                </div>

                {result.extraction_details && (
                  <div className="mt-4 p-4 bg-white rounded-lg">
                    <p className="font-semibold text-gray-900 mb-2">Детали извлечения:</p>
                    <div className="grid grid-cols-2 gap-2 text-sm">
                      <div>
                        <span className="text-gray-600">Видео:</span>
                        <span className={`ml-2 ${result.extraction_details.video.success ? 'text-green-600' : 'text-red-600'}`}>
                          {result.extraction_details.video.success ? '✓ Найдено' : '✗ Не найдено'}
                        </span>
                      </div>
                      <div>
                        <span className="text-gray-600">Аудио:</span>
                        <span className={`ml-2 ${result.extraction_details.audio.success ? 'text-green-600' : 'text-red-600'}`}>
                          {result.extraction_details.audio.success ? '✓ Найдено' : '✗ Не найдено'}
                        </span>
                      </div>
                      <div className="col-span-2">
                        <span className="text-gray-600">Уверенность:</span>
                        <span className="ml-2 font-semibold">
                          {result.extraction_details.confidence === 'high' ? 'Высокая' : 
                           result.extraction_details.confidence === 'medium' ? 'Средняя' : 'Низкая'}
                        </span>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Leak Reports Table */}
      <div className="bg-white rounded-xl shadow-lg p-6">
        <h2 className="text-xl font-bold text-gray-900 mb-4">Отчёты об утечках</h2>
        
        {loading ? (
          <div className="text-center py-8">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          </div>
        ) : reports.length === 0 ? (
          <div className="text-center py-8 text-gray-500">
            <FiEye className="w-12 h-12 mx-auto mb-2 text-gray-300" />
            <p>Отчётов об утечках пока нет</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b-2 border-gray-200">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">ID</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Видео</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Подозреваемый</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Водяной знак</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Статус</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Дата</th>
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Действия</th>
                </tr>
              </thead>
              <tbody>
                {reports.map((report) => (
                  <tr key={report.id} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4">#{report.id}</td>
                    <td className="py-3 px-4">{report.video_title}</td>
                    <td className="py-3 px-4">{report.suspected_user_name}</td>
                    <td className="py-3 px-4 font-mono text-sm">{report.watermark_id}</td>
                    <td className="py-3 px-4">
                      <span className={`px-2 py-1 rounded-full text-xs font-semibold ${getStatusColor(report.status)}`}>
                        {getStatusLabel(report.status)}
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm">
                      {new Date(report.reported_at).toLocaleDateString('ru-RU')}
                    </td>
                    <td className="py-3 px-4">
                      {report.status === 'pending' && (
                        <div className="flex space-x-2">
                          <button
                            onClick={() => handleUpdateReport(report.id, 'confirmed', true)}
                            className="px-3 py-1 bg-red-500 text-white text-xs rounded hover:bg-red-600"
                            title="Подтвердить и заблокировать"
                          >
                            Подтвердить и бан
                          </button>
                          <button
                            onClick={() => handleUpdateReport(report.id, 'false_positive')}
                            className="px-3 py-1 bg-green-500 text-white text-xs rounded hover:bg-green-600"
                            title="Отметить как ложное"
                          >
                            Ложное
                          </button>
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

export default LeakDetection;
