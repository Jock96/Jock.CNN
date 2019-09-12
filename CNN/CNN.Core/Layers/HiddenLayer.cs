namespace CNN.Core.Layers
{
    using CNN.Core.Models;

    using System.Collections.Generic;
    using System;
    using CNN.BL.Enums;
    using CNN.Core.Extensions;
    using CNN.BL.Constants;
    using CNN.BL.Helpers;

    /// <summary>
    /// Класс скрытого слоя.
    /// </summary>
    public class HiddenLayer : Layer
    {
        /// <summary>
        /// Список нейронов скрытого слоя.
        /// </summary>
        private List<NeuronModel> _convolutionalLayerData;

        /// <summary>
        /// Данные скрытого слоя.
        /// </summary>
        private List<NeuronModel> _hiddenLayerData;

        /// <summary>
        /// Тип слоя.
        /// </summary>
        public override LayerType LayerType => LayerType.Hidden;

        /// <summary>
        /// Класс скрытого слоя.
        /// </summary>
        /// <param name="convolutionalLayer">Данные свёрточного слоя.</param>
        public HiddenLayer(List<NeuronModel> convolutionalLayer)
        {
            _convolutionalLayerData = convolutionalLayer;
        }

        /// <summary>
        /// Приготовить к распознаванию.
        /// </summary>
        /// <param name="neuronIndexToWeightsValueDictionary">Словарь значений весов,
        /// где ключ - индекс нейрона, значение - веса нейрона.</param>
        public void RecognizeMode(Dictionary<int, List<double>> neuronIndexToWeightsValueDictionary)
        {
            _hiddenLayerData = new List<NeuronModel>();
            var countOfNeuronsInHiddenLayer = (int)_convolutionalLayerData.Count / 2;

            var inputs = new List<double>();

            _convolutionalLayerData.ForEach(neuronOfConvolutionalLayer
                => inputs.Add(neuronOfConvolutionalLayer.Output));

            for (var index = 0; index < countOfNeuronsInHiddenLayer; ++index)
            {
                if (!neuronIndexToWeightsValueDictionary.TryGetValue(index, out var weights))
                    ErrorHelper.GetDataError();

                var neuron = new NeuronModel(inputs, weights);
                _hiddenLayerData.Add(neuron);
            }
        }

        /// <summary>
        /// Инициализация слоя.
        /// </summary>
        public void Initialize()
        {
            _hiddenLayerData = new List<NeuronModel>();
            var countOfNeuronsInHiddenLayer = (int)_convolutionalLayerData.Count / 2;

            for (var index = 0; index < countOfNeuronsInHiddenLayer; ++index)
                _hiddenLayerData.Add(CreateNeuronByData());
        }

        /// <summary>
        /// Возвращает нейроны скрытого слоя.
        /// </summary>
        /// <returns></returns>
        public List<NeuronModel> GetLayerNeurons() => _hiddenLayerData;

        /// <summary>
        /// Создать нейрон по имеющимся данным.
        /// </summary>
        /// <returns>Возвращает нейрон.</returns>
        private NeuronModel CreateNeuronByData()
        {
            var inputs = new List<double>();
            var weights = new List<double>();

            var emptyLastWeights = new List<double>();

            foreach (var neuron in _convolutionalLayerData)
            {
                inputs.Add(neuron.Output);
                weights.Add(GetInitializedWeight());

                emptyLastWeights.Add(0d);
            }

            var hiddenLayerNeuron = new NeuronModel(inputs, weights)
            {
                LastWeights = emptyLastWeights
            };

            return hiddenLayerNeuron;
        }

        /// <summary>
        /// Инициализация случайных весов.
        /// </summary>
        /// <returns>Возвращает случайный вес.</returns>
        private double GetInitializedWeight() => new Random().NextDouble();

        #region Обновление значений нейронов.

        /// <summary>
        /// Обновление дельт нейронов.
        /// </summary>
        /// <param name="deltas">Список дельт.</param>
        public void UpdateDeltas(List<double> deltas) => 
            _hiddenLayerData.ForEach(neuron =>
            neuron.Delta = deltas[_hiddenLayerData.IndexOf(neuron)]);

        /// <summary>
        /// Обновление весов нейронов на слое.
        /// </summary>
        /// <param name="neuronIndexToWeightsDictionary">
        /// Словарь обновлённых весов (индекс нейрона/новые веса).</param>
        public void UpdateLayerNeuronsWeights(
            Dictionary<int, List<double>> neuronIndexToWeightsDictionary)
        {
            foreach (var neuronKey in neuronIndexToWeightsDictionary.Keys)
            {
                if(!neuronIndexToWeightsDictionary.TryGetValue(neuronKey, out var updatedWeights))
                {
                    var exception = new Exception("не валидные данные в словаре обновлённых весов.");

                    Console.WriteLine(ConsoleMessageConstants.ERROR_MESSAGE + exception.ToString());

                    Console.ReadKey();
                }

                _hiddenLayerData[neuronKey].UpdateWeights(updatedWeights);
            }
        }

        /// <summary>
        /// Задать новые входные значения на слой.
        /// </summary>
        /// <param name="inputs"></param>
        public void UpdateNeuronsInputs(List<double> inputs) =>
            _hiddenLayerData.ForEach(neuron => neuron.Inputs = inputs);

        #endregion
    }
}
