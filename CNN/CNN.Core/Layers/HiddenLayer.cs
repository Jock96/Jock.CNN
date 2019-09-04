namespace CNN.Core.Layers
{
    using CNN.Core.Models;

    using System.Collections.Generic;
    using System;
    using CNN.BL.Enums;

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

            foreach (var neuron in _convolutionalLayerData)
            {
                inputs.Add(neuron.Output);
                weights.Add(GetInitializedWeight());
            }

            var hiddenLayerNeuron = new NeuronModel(inputs, weights);
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

        #endregion
    }
}
