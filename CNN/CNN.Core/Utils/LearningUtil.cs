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

        // Метод обучения (эпохи и итерации)

        public void StartToLearn()
        {
            var errorValue = 0d;

            for (var epochIndex = 0; epochIndex < _configuration.EpochCount; ++epochIndex)
            {
                for (var iteration = 0; iteration < _configuration.IterationsInEpochCount; ++iteration)
                {
                    // Обратное распространение.

                    Backpropagation();

                    var outputLayer = (OutputLayer)_layers.Find(layer => layer.LayerType.Equals(LayerType.Output));
                    var outputValue = outputLayer.GetOutputNeuron().Output;

                    errorValue = ErrorByRootMSE(epochIndex, outputValue);

                    GetOutputCallback(outputValue, errorValue, epochIndex, iteration);
                }
            }
        }

        #region Расчёты по методу обрабтного распространения.

        private void Backpropagation()
        {
            var outputLayer = (OutputLayer)_layers.Find(layer => layer.LayerType.Equals(LayerType.Output));
            var hiddenLayer = (HiddenLayer)_layers.Find(layer => layer.LayerType.Equals(LayerType.Hidden));
            var convolutionalLayer = (ConvolutionalLayer)_layers.Find(layer => layer.LayerType.Equals(LayerType.Convolutional));
            var inputLayer = (InputLayer)_layers.Find(layer => layer.LayerType.Equals(LayerType.Input));

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

            // Получение градиента между нейронами скрытого слоя и выходного, и обновление весов.

            // Для свёртончного - 
            // подсчитывается общая ошибка для каждого значения
            // в ядре фильтра (в общем их 9 у меня), 
            // К каждому из ядра прибавляется общая сумма
            // соответствующей ячейки в своей позиции
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
        private double GetHiddenLayerNeuronDelta(double neuronOutput, int indexInLayer, NeuronModel outputNeuron) => 
            DerivativeActivationFunction(neuronOutput) * outputNeuron.Inputs[indexInLayer] * outputNeuron.Delta;

        /// <summary>
        /// Производная от функции активации (сигмоид).
        /// </summary>
        /// <param name="neuronOutput">Вывод нейрона.</param>
        /// <returns>Вовзращает результат производной функции активации (сигмоид) нейрона.</returns>
        private double DerivativeActivationFunction(double neuronOutput) => 
            ((1 - neuronOutput) * neuronOutput);

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

            Console.WriteLine($"{ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE}");
            Console.ReadKey();
        }

        #endregion
    }
}
