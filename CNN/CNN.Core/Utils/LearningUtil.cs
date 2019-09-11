namespace CNN.Core.Utils
{
    using System;
    using System.Collections.Generic;

    using BL.Enums;
    using BL.Constants;

    using CNN.Core.Layers;
    using CNN.Core.Models;

    /// <summary>
    /// Инструмент обучения нейронной сети.
    /// </summary>
    public class LearningUtil
    {
        /// <summary>
        /// Список слоёв нейронной сети.
        /// </summary>
        private List<Layer> _layers;

        /// <summary>
        /// Конфигурация нейронной сети.
        /// </summary>
        private Configuration _configuration;

        /// <summary>
        /// Суммарная ошибка по эпохам.
        /// </summary>
        private double _errorSummary;

        /// <summary>
        /// Инструмент обучения нейронной сети.
        /// </summary>
        /// <param name="layers">Список слоёв нейронной сети.</param>
        /// <param name="configuration">Конфигурация нейронной сети.</param>
        public LearningUtil(List<Layer> layers, Configuration configuration)
        {
            _layers = layers;
            _configuration = configuration;
        }

        /// <summary>
        /// Инициализация обучения нейронной сети.
        /// </summary>
        public void StartToLearn()
        {
            var errorValue = 0d;

            for (var epochIndex = 0; epochIndex < _configuration.EpochCount; ++epochIndex)
            {
                for (var iteration = 0; iteration < _configuration.IterationsInEpochCount; ++iteration)
                {
                    var outputLayer = (OutputLayer)_layers.Find(layer =>
                    layer.LayerType.Equals(LayerType.Output));

                    var hiddenLayer = (HiddenLayer)_layers.Find(layer =>
                    layer.LayerType.Equals(LayerType.Hidden));

                    var convolutionalLayer = (ConvolutionalLayer)_layers.Find(layer =>
                    layer.LayerType.Equals(LayerType.Convolutional));

                    var inputLayer = (InputLayer)_layers.Find(layer =>
                    layer.LayerType.Equals(LayerType.Input));

                    var outputValue = outputLayer.GetOutputNeuron().Output;
                    errorValue = ErrorByRootMSE(epochIndex, outputValue);

                    Backpropagation(inputLayer, convolutionalLayer, hiddenLayer, outputLayer);
                    GetOutputCallback(outputValue, errorValue, epochIndex, iteration);

                    LayersUpdate(outputLayer, hiddenLayer, convolutionalLayer, inputLayer);
                }
            }

            Console.WriteLine($"{ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE}");
            Console.ReadKey();
        }

        /// <summary>
        /// Обновить слои.
        /// </summary>
        /// <param name="outputLayer">Выходной слой.</param>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        /// <param name="convolutionalLayer">Свёрточный слой.</param>
        /// <param name="inputLayer">Входной слой.</param>
        private void LayersUpdate(OutputLayer outputLayer, HiddenLayer hiddenLayer, ConvolutionalLayer convolutionalLayer, InputLayer inputLayer)
        {
            if (_configuration.IterationsInEpochCount > 1)
            {
                // Задать новые значения входного слоя.
            }

            convolutionalLayer.UpdateData(FilterCoreModel.GetCore, inputLayer.GetLayerNeurons());

            var inputsToHiddenLayer = new List<double>();

            convolutionalLayer.GetLayerNeurons().ForEach(neuron =>
            inputsToHiddenLayer.Add(neuron.Output));

            hiddenLayer.UpdateNeuronsInputs(inputsToHiddenLayer);

            var inputsToOutputLayer = new List<double>();

            hiddenLayer.GetLayerNeurons().ForEach(neuron =>
            inputsToOutputLayer.Add(neuron.Output));

            outputLayer.UpdateNeuronInputs(inputsToOutputLayer);
        }

        #region Расчёты по методу обратного распространения.

        /// <summary>
        /// Метод обратного распространения ошибки.
        /// </summary>
        /// <param name="inputLayer">Входной слой.</param>
        /// <param name="convolutionalLayer">Свёрточный слой.</param>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        /// <param name="outputLayer">Выходной слой.</param>
        private void Backpropagation(InputLayer inputLayer, ConvolutionalLayer convolutionalLayer,
            HiddenLayer hiddenLayer, OutputLayer outputLayer)
        {
            HiddenToOutputDeltasWork(outputLayer, hiddenLayer);
            HiddentToOutputWeightsWork(outputLayer, hiddenLayer);

            ConvolutionalToHiddenDeltasWork(hiddenLayer, convolutionalLayer);
            ConvolutionalToHiddenWeightsWork(hiddenLayer, convolutionalLayer);

            // TODO: В ядре всё по нулям.
            FilterCoreWork(convolutionalLayer, inputLayer);
        }

        /// <summary>
        /// Выполнить вычисления для ядра фильтра.
        /// </summary>
        /// <param name="convolutionalLayer">Свёрточный слой.</param>
        /// <param name="inputLayer">Входной слой.</param>
        private void FilterCoreWork(ConvolutionalLayer convolutionalLayer, InputLayer inputLayer)
        {
            var inputLayerNeurons = inputLayer.GetLayerNeurons();
            var convolutionalLayerDeltas = new List<double>();

            convolutionalLayer.GetLayerNeurons().ForEach(neuron => convolutionalLayerDeltas.Add(neuron.Delta));

            var filterCoreMatrixOfMiddleDeltas = GetFilterCoreMiddleDeltasMatrix(
                inputLayerNeurons, convolutionalLayerDeltas);

            var filterCoreMatrixOfMiddleGradients = GetFilterCoreMiddleGradientMatrix(
                filterCoreMatrixOfMiddleDeltas, inputLayerNeurons);

            UpdateCore(filterCoreMatrixOfMiddleGradients);
        }

        /// <summary>
        /// Обновить ядро фильтра.
        /// </summary>
        /// <param name="gradients">Матрица градиентов.</param>
        private void UpdateCore(double[,] gradients)
        {
            var updatedCore = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                MatrixConstants.FILTER_MATRIX_SIZE];

            for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                {
                    updatedCore[xIndex, yIndex] = _configuration.Epsilon *
                        gradients[xIndex, yIndex] + _configuration.Alpha *
                        FilterCoreModel.LastCoreValue[xIndex, yIndex];
                }

            FilterCoreModel.UpdateCore(updatedCore);
        }

        /// <summary>
        /// Получить матрицу средних градиентов.
        /// </summary>
        /// <param name="filterCoreMatrixOfMiddleDeltas">Матрицасредних дельт.</param>
        /// <param name="inputLayerData">Данные входного слоя.</param>
        /// <returns>Возвращает матрицу средних градиентов.</returns>
        private double[,] GetFilterCoreMiddleGradientMatrix(
            double[,] filterCoreMatrixOfMiddleDeltas, Dictionary<string, double> inputLayerData)
        {
            var middleGradientMatrix = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                MatrixConstants.FILTER_MATRIX_SIZE];

            for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                {
                    var valuesFromInputData = GetMatrixOfValuesFromInputData(inputLayerData,
                        xIndex, yIndex);

                    var summary = 0d;
                    foreach (var value in valuesFromInputData)
                    {
                        summary += value;
                    }

                    var middleGradient = filterCoreMatrixOfMiddleDeltas[xIndex, yIndex] * summary;
                }

            return middleGradientMatrix;
        }

        /// <summary>
        /// Получить значения средних дельт для ядра фильтра.
        /// </summary>
        /// <param name="inputLayerData">Данные входного слоя.</param>
        /// <param name="convolutionalLayerDeltas">Дельты свёрточного слоя.</param>
        /// <returns>Возвращает матрицу средньких дельт для ядра фильтра.</returns>
        private double[,] GetFilterCoreMiddleDeltasMatrix(Dictionary<string, double> inputLayerData,
            List<double> convolutionalLayerDeltas)
        {
            var middleDeltasMatrix = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                MatrixConstants.FILTER_MATRIX_SIZE];

            var countOfNeurons = Math.Pow(MatrixConstants.FILTER_MATRIX_SIZE, 2);
            var summaryOfConvolutionalLayerDeltas = 0d;

            convolutionalLayerDeltas.ForEach(delta => summaryOfConvolutionalLayerDeltas += delta);

            for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                {
                    var valuesFromInputData = GetMatrixOfValuesFromInputData(inputLayerData,
                        xIndex, yIndex);

                    foreach (var output in valuesFromInputData)
                    {
                        var middleDelta = DerivativeActivationFunction(output) * 
                            summaryOfConvolutionalLayerDeltas * 
                            FilterCoreModel.GetCore[xIndex, yIndex];

                        middleDeltasMatrix[xIndex, yIndex] = middleDelta;
                    }
                }

            return middleDeltasMatrix;
        }

        /// <summary>
        /// Получить матрицу значений из входного слоя.
        /// </summary>
        /// <param name="inputLayerData">Значения входного слоя.</param>
        /// <param name="xPositionInFilterCore">Позиция по X.</param>
        /// <param name="yPositionInFilterCore">Позиция по Y.</param>
        /// <returns></returns>
        private double[,] GetMatrixOfValuesFromInputData(Dictionary<string, double> inputLayerData,
            int xPositionInFilterCore, int yPositionInFilterCore)
        {
            var valuesFromInputData = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                        MatrixConstants.FILTER_MATRIX_SIZE];

            for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                {
                    var locationString = $"{MatrixConstants.POSITION_IN_X_AXIS}" +
                        $"{xPositionInFilterCore + xIndex}" +
                        $"{MatrixConstants.KEY_SEPARATOR}" +
                        $"{MatrixConstants.POSITION_IN_Y_AXIS}" +
                        $"{yPositionInFilterCore + yIndex}";

                    inputLayerData.TryGetValue(locationString, out var data);
                    valuesFromInputData[xIndex, yIndex] = data;
                }

            return valuesFromInputData;
        }

        /// <summary>
        /// Получение градиента между нейронами свёрточного слоя и скрытого, и обновление весов.
        /// </summary>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        /// <param name="convolutionalLayer">Свёрточный слой.</param>
        private void ConvolutionalToHiddenWeightsWork(HiddenLayer hiddenLayer, ConvolutionalLayer convolutionalLayer)
        {
            var convolutionalLayerNeurons = convolutionalLayer.GetLayerNeurons();
            var convolutionalLayerOutputs = new List<double>();

            var hiddenLayerNeurons = hiddenLayer.GetLayerNeurons();
            var hiddenLayerDeltas = new List<double>();

            convolutionalLayerNeurons.ForEach(neuron => convolutionalLayerOutputs.Add(neuron.Output));
            hiddenLayerNeurons.ForEach(neuron => hiddenLayerDeltas.Add(neuron.Delta));

            var convolutionalNeuronToGradientsDictionary = GetConvolutionalToHiddenGradients(
                convolutionalLayerOutputs, hiddenLayerDeltas);

            UpdateConvolutionalToHiddenWeights(convolutionalNeuronToGradientsDictionary, hiddenLayer);
        }

        /// <summary>
        /// Получение и обновление дельты свёрточного слоя.
        /// </summary>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        /// <param name="convolutionalLayer">Свёрточный слой.</param>
        private void ConvolutionalToHiddenDeltasWork(HiddenLayer hiddenLayer, ConvolutionalLayer convolutionalLayer)
        {
            var convolutionalLayerNeurons = convolutionalLayer.GetLayerNeurons();
            var convolutionalLayerOutputs = new List<double>();

            convolutionalLayerNeurons.ForEach(neuron =>
            convolutionalLayerOutputs.Add(neuron.Output));

            var convolutionalLayerDeltas = GetConvolutionalLayerDeltas(convolutionalLayerOutputs,
                hiddenLayer.GetLayerNeurons());

            convolutionalLayer.UpdateDeltas(convolutionalLayerDeltas);
        }

        /// <summary>
        /// Получение градиента между нейронами скрытого слоя и выходного, и обновление весов.
        /// </summary>
        /// <param name="outputLayer">Выходной слой.</param>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        private void HiddentToOutputWeightsWork(OutputLayer outputLayer, HiddenLayer hiddenLayer)
        {
            var hiddenLayerOutputs = new List<double>();
            hiddenLayer.GetLayerNeurons().ForEach(neuron => hiddenLayerOutputs.Add(neuron.Output));

            var hiddenLayerGradients = GetHiddenToOutputGradients(hiddenLayerOutputs,
                outputLayer.GetOutputNeuron().Output);

            UpdateHiddenToOutputWeights(hiddenLayerGradients, outputLayer);
        }

        /// <summary>
        /// Вычисление дельт выходного и скрытого слоя.
        /// </summary>
        /// <param name="outputLayer">Выходной слой.</param>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        private void HiddenToOutputDeltasWork(OutputLayer outputLayer, HiddenLayer hiddenLayer)
        {
            // Получение и обновление дельты выходного слоя.

            var outputNeuron = outputLayer.GetOutputNeuron();

            var deltaOfOutputNeuron = GetOutputLayerNeuronDelta(outputNeuron.Output);
            outputLayer.UpdateDeltaOfNeuronInLayer(deltaOfOutputNeuron);

            // Получение и обновление дельты скрытого слоя.

            var hiddenLayerNeurons = hiddenLayer.GetLayerNeurons();

            var deltasOfHiddenLayerNeurons = new List<double>();

            foreach (var neuron in hiddenLayerNeurons)
            {
                var index = hiddenLayerNeurons.IndexOf(neuron);
                var delta = GetHiddenLayerNeuronDelta(neuron.Output, index, outputNeuron);

                deltasOfHiddenLayerNeurons.Add(delta);
            }

            hiddenLayer.UpdateDeltas(deltasOfHiddenLayerNeurons);
        }

        /// <summary>
        /// Получить словарь, где ключ - индекс нейрона свёрточного слоя,
        /// значение - список градиентов данного нейрона к нейронам скрытого слоя.
        /// </summary>
        /// <param name="outputs">Выходы нейронов свёрточного слоя.</param>
        /// <param name="deltas">Дельты скрытого слоя.</param>
        /// <returns>Возвращает словарь (индекс/градиенты).</returns>
        private Dictionary<int, List<double>> GetConvolutionalToHiddenGradients(
            List<double> outputs, List<double> deltas)
        {
            var indexToGradientsDictionary = new Dictionary<int, List<double>>();
            var indexOfNeuron = 0;

            foreach (var output in outputs)
            {
                var gradients = new List<double>();

                foreach (var delta in deltas)
                {
                    var gradient = delta * output;
                    gradients.Add(gradient);
                }

                indexToGradientsDictionary.Add(indexOfNeuron, gradients);
                ++indexOfNeuron;
            }

            return indexToGradientsDictionary;
        }

        /// <summary>
        /// Получить все дельты свёрточного слоя.
        /// </summary>
        /// <param name="outputs">Выходы нейронов свёрточного слоя.</param>
        /// <param name="hiddenLayerNeurons">Нейроны скрытого слоя.</param>
        /// <returns>Возвращает список дельт свёрточного слоя.</returns>
        private List<double> GetConvolutionalLayerDeltas(
            List<double> outputs, List<NeuronModel> hiddenLayerNeurons)
        {
            var convolutionalLayerDeltas = new List<double>();
            var indexOfCurrentNeuronInConvolutionalLayer = 0;

            foreach (var output in outputs)
            {
                var sum = 0d;

                hiddenLayerNeurons.ForEach(neuron => 
                sum += neuron.Weights[indexOfCurrentNeuronInConvolutionalLayer] * neuron.Delta);

                var delta = DerivativeActivationFunction(output) * sum;
                convolutionalLayerDeltas.Add(delta);

                ++indexOfCurrentNeuronInConvolutionalLayer;
            }

            return convolutionalLayerDeltas;
        }

        /// <summary>
        /// Обновить веса между скрытым и выходным слоем.
        /// </summary>
        /// <param name="gradients">Список градиентов между скрытым и выходным слоями.</param>
        /// <param name="outputLayer">Выходной слой.</param>
        private void UpdateHiddenToOutputWeights(List<double> gradients, OutputLayer outputLayer)
        {
            var updatedWeights = new List<double>();
            var indexOfGradient = 0;

            foreach (var gradient in gradients)
            {
                var lastWeight = outputLayer.GetOutputNeuron().
                    LastWeights[indexOfGradient];

                var updatedWeight = _configuration.Epsilon * gradient + 
                    _configuration.Alpha * lastWeight;

                updatedWeights.Add(updatedWeight);
                ++indexOfGradient;
            }

            outputLayer.UpdateWeightsOfNeuronInLayer(updatedWeights);
        }

        /// <summary>
        /// Обновить веса между свёрточным и скрытым слоями.
        /// </summary>
        /// <param name="gradients">Словарь градиентов (индекс нейрона свёрточного слоя/список градиентов).</param>
        /// <param name="hiddenLayer">Скрытый слой.</param>
        private void UpdateConvolutionalToHiddenWeights(
            Dictionary<int, List<double>> gradients, HiddenLayer hiddenLayer)
        {
            var neuronIndexToUpdatedWeightsDictionary = new Dictionary<int, List<double>>();
            var hiddenLayerNeurons = hiddenLayer.GetLayerNeurons();

            var neuronIndex = 0;
            foreach (var neuron in hiddenLayerNeurons)
            {
                var updatedWeights = new List<double>();

                var weightIndex = 0;

                foreach (var weights in neuron.Weights)
                {
                    var gradient = gradients[weightIndex][neuronIndex];

                    var updatedWeight = _configuration.Epsilon * gradient +
                        _configuration.Alpha * neuron.LastWeights[weightIndex];

                    updatedWeights.Add(updatedWeight);

                    ++weightIndex;
                }

                neuronIndexToUpdatedWeightsDictionary.Add(neuronIndex, updatedWeights);
                ++neuronIndex;
            }

            hiddenLayer.UpdateLayerNeuronsWeights(neuronIndexToUpdatedWeightsDictionary);
        }

        /// <summary>
        /// Получить список всех градиентов между скрытым и выходным слоем.
        /// </summary>
        /// <param name="outputs">Выходы нейронов скрытого слоя.</param>
        /// <param name="outputDelta">Дельта выходного нейрона.</param>
        /// <returns>Возвращает список градиентов для скрытого слоя.</returns>
        private List<double> GetHiddenToOutputGradients(List<double> outputs, double outputDelta)
        {
            var gradientsList = new List<double>();

            outputs.ForEach(output => gradientsList.Add(output * outputDelta));

            return gradientsList;
        }

        /// <summary>
        /// Расчёт дельты выходного нейрона.
        /// </summary>
        /// <param name="neuronOutput">Вывод нейрона.</param>
        /// <returns>Возвращает дельту выходного нейрона.</returns>
        private double GetOutputLayerNeuronDelta(double neuronOutput) => 
            (_configuration.IdealResult - neuronOutput) * DerivativeActivationFunction(neuronOutput);

        /// <summary>
        /// Расчёт дельты для нейронов скрытого слоя.
        /// </summary>
        /// <param name="neuronOutput">Вывод нейрона.</param>
        /// <param name="indexInLayer">Индекс в скрытом слое.</param>
        /// <param name="outputNeuron">Нейрон выходного слоя.</param>
        /// <returns></returns>
        private double GetHiddenLayerNeuronDelta(
            double neuronOutput, int indexInLayer, NeuronModel outputNeuron) => 
            DerivativeActivationFunction(neuronOutput) *
            outputNeuron.Inputs[indexInLayer] * outputNeuron.Delta;

        /// <summary>
        /// Производная от функции активации (сигмоид).
        /// </summary>
        /// <param name="neuronOutput">Вывод нейрона.</param>
        /// <returns>Вовзращает результат производной функции активации (сигмоид) нейрона.</returns>
        private double DerivativeActivationFunction(double neuronOutput) =>
            ((1 - neuronOutput) * neuronOutput);

#if !DEBUG && !RELEASE

        /// <summary>
        /// Производная от функции активации (Гиперболический тангенс).
        /// </summary>
        /// <param name="neuronOutput">Вывод нейрона.</param>
        /// <returns>Вовзращает результат производной функции активации (сигмоид) нейрона.</returns>
        private double DerivativeActivationFunction(double neuronOutput) =>
            (1 - Math.Pow(neuronOutput, 2));

#endif

        #endregion

        #region Информационные команды.

        /// <summary>
        /// Просчёт ошибки.
        /// </summary>
        /// <param name="currentEpoch">Текущая эпоха.</param>
        /// <param name="currentOutput">Текущий вывод на конечном слое.</param>
        /// <returns>Возвращает значение ошибки.</returns>
        private double ErrorByRootMSE(int currentEpoch, double currentOutput)
        {
            _errorSummary += Math.Pow(_configuration.IdealResult - currentOutput, 2);

            return Math.Pow(_errorSummary / currentEpoch, 0.5);
        }

        /// <summary>
        /// Дать обратную связь по информации текущей итерациии эпохи.
        /// </summary>
        /// <param name="outputValue">Значение выходного слоя.</param>
        /// <param name="errorValue">Значение ошибки.</param>
        /// <param name="currentEpoch">Текущая эпоха.</param>
        /// <param name="currentItaration">Текущая итерация.</param>
        private void GetOutputCallback(double outputValue, double errorValue,
            int currentEpoch, int currentItaration)
        {
            var currentEpochMessage = $"Текущая эпоха - {currentEpoch + 1} ";

            var currentIterationMessage = $"(итерация {currentItaration + 1} " +
                $"из {_configuration.IterationsInEpochCount}).";

            var outputValueMessage = $"Значение выходного слоя {outputValue}.";

            var errorValueMessage = $"Значение ошибки {errorValue}.";

            Console.WriteLine($"{ConsoleMessageConstants.INFO_MESSAGE}\n" +
                $"{currentEpochMessage}{currentIterationMessage}\n" +
                $"{outputValueMessage}\n" +
                $"{errorValueMessage}");
        }

#endregion
    }
}
