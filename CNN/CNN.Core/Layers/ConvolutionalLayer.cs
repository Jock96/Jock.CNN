namespace CNN.Core.Layers
{
    using System.Collections.Generic;
    using System;

    using CNN.BL.Constants;

    using Models;
    using CNN.BL.Enums;

    /// <summary>
    /// Класс свёрточного слоя.
    /// </summary>
    public class ConvolutionalLayer : Layer
    {
        /// <summary>
        /// Данные входного слоя.
        /// </summary>
        private Dictionary<string, double> _inputLayerData;

        /// <summary>
        /// Данные свёрточного слоя.
        /// </summary>
        private List<NeuronModel> _convolutionalLayerData;

        /// <summary>
        /// Тип слоя.
        /// </summary>
        public override LayerType LayerType => LayerType.Convolutional;

        /// <summary>
        /// Класс свёрточного слоя.
        /// </summary>
        /// <param name="inputLayerData">Данные входного слоя.</param>
        public ConvolutionalLayer(Dictionary<string, double> inputLayerData)
        {
            _inputLayerData = inputLayerData;
        }

        /// <summary>
        /// Инициализацяи свёрточного слоя.
        /// </summary>
        /// <param name="filterCore">Ядро фильтра.</param>
        public void Initialize(double[,] filterCore)
        {
            _convolutionalLayerData = new List<NeuronModel>();

            var offset = MatrixConstants.MATRIX_SIZE - MatrixConstants.FILTER_MATRIX_SIZE;
            var step = MatrixConstants.MATRIX_SIZE - offset;

            var emptyLastWeights = new List<double>();

            for (var index = 0; index < MatrixConstants.MATRIX_SIZE *
                MatrixConstants.MATRIX_SIZE; ++index)
                emptyLastWeights.Add(0d);

            for (var xIndex = 0; xIndex < step; ++xIndex)
                for (var yIndex = 0; yIndex < step; ++yIndex)
                {
                    var inputs = new List<double>();
                    var weights = new List<double>();

                    GetDataByFilterCore(filterCore, xIndex, yIndex, inputs, weights);
                    var neuron = new NeuronModel(inputs, weights)
                    {
                        LastWeights = emptyLastWeights
                    };

                    _convolutionalLayerData.Add(neuron);
                }
        }

        /// <summary>
        /// Получить значение нейронов свёрточного слоя.
        /// </summary>
        /// <returns>Возвращает нейроны свёрточного слоя.</returns>
        public List<NeuronModel> GetLayerNeurons() => _convolutionalLayerData;

        /// <summary>
        /// Получение данных по ядру фильтра.
        /// </summary>
        /// <param name="filterCore">Ядро фильтра.</param>
        /// <param name="xIndex">Индекс матрицы по X.</param>
        /// <param name="yIndex">Индекс матрицы по Y.</param>
        /// <param name="inputs">Список входных данных для заполнения.</param>
        /// <param name="weights">Список весов для заполнения.</param>
        private void GetDataByFilterCore(double[,] filterCore, int xIndex, int yIndex, List<double> inputs, List<double> weights)
        {
            for (var xCoreIndex = 0; xCoreIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xCoreIndex)
                for (var yCoreIndex = 0; yCoreIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yCoreIndex)
                {
                    var comparsionString = $"{MatrixConstants.POSITION_IN_X_AXIS}{xIndex}" +
                        $"{MatrixConstants.KEY_SEPARATOR}" +
                        $"{MatrixConstants.POSITION_IN_Y_AXIS}{yIndex}";

                    // TODO: Исправить ошибку в заполнении слоя.

                    if (!_inputLayerData.TryGetValue(comparsionString, out var inputValue))
                    {
                        var valueException = new Exception("Не удалось получить значение.");
                        Console.WriteLine(ConsoleMessageConstants.ERROR_MESSAGE + valueException);
                    }

                    inputs.Add(inputValue);
                    weights.Add(filterCore[xCoreIndex, yCoreIndex]);
                }
        }

        #region Обновление значений нейронов.

        /// <summary>
        /// Обновление дельт свёрточного слоя.
        /// </summary>
        /// <param name="deltas">Список дельт.</param>
        public void UpdateDeltas(List<double> deltas) =>
            _convolutionalLayerData.ForEach(neuron =>
            neuron.Delta = deltas[_convolutionalLayerData.IndexOf(neuron)]);

        /// <summary>
        /// Обновить данные слоя.
        /// </summary>
        /// <param name="updatedFilterCore">Обновлённое ядро фильтра.</param>
        /// <param name="updatedInputLayerData">Обновлённый входной слой.</param>
        public void UpdateData(double[,] updatedFilterCore,
            Dictionary<string, double> updatedInputLayerData)
        {
            var offset = MatrixConstants.MATRIX_SIZE - MatrixConstants.FILTER_MATRIX_SIZE;
            var step = MatrixConstants.MATRIX_SIZE - offset;

            var updatedConvolutionalLayerNeurons = new List<NeuronModel>();
            var neuronIndex = 0;

            for (var xIndex = 0; xIndex < step; ++xIndex)
                for (var yIndex = 0; yIndex < step; ++yIndex)
                {
                    var inputs = new List<double>();
                    var weights = new List<double>();

                    GetDataByFilterCore(updatedFilterCore, xIndex, yIndex, inputs, weights);

                    var neuron = new NeuronModel(inputs, weights)
                    {
                        LastWeights = _convolutionalLayerData[neuronIndex].LastWeights
                    };

                    ++neuronIndex;
                }
        }

        #endregion
    }
}
