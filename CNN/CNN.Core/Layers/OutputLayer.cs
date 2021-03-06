﻿namespace CNN.Core.Layers
{
    using System;
    using System.Collections.Generic;
    using CNN.BL.Enums;
    using Models;
    using CNN.Core.Extensions;
    using CNN.BL.Helpers;

    /// <summary>
    /// Класс выходного слоя.
    /// </summary>
    public class OutputLayer : Layer
    {
        /// <summary>
        /// Значения скрытого слоя.
        /// </summary>
        private List<NeuronModel> _hiddenLayerData;

        /// <summary>
        /// Значение выходного нейрона.
        /// </summary>
        private NeuronModel _outputNeuron;

        /// <summary>
        /// Тип слоя.
        /// </summary>
        public override LayerType LayerType => LayerType.Output;

        /// <summary>
        /// Класс выходного слоя.
        /// </summary>
        /// <param name="hiddenLayerData">Значения скрытого слоя.</param>
        public OutputLayer(List<NeuronModel> hiddenLayerData)
        {
            _hiddenLayerData = hiddenLayerData;
        }

        /// <summary>
        /// Приготовить к распознаванию.
        /// </summary>
        /// <param name="neuronIndexToWeightsValueDictionary">Словарь значений весов,
        /// где ключ - индекс нейрона, значение - веса нейрона.</param>
        public void RecognizeMode(Dictionary<int, List<double>> neuronIndexToWeightsValueDictionary)
        {
            var inputs = new List<double>();

            if (!neuronIndexToWeightsValueDictionary.TryGetValue(0, out var weights))
                ErrorHelper.GetDataError();

            _hiddenLayerData.ForEach(neuron => inputs.Add(neuron.Output));

            _outputNeuron = new NeuronModel(inputs, weights);
        }

        /// <summary>
        /// Инициализация слоя.
        /// </summary>
        public void Initilize()
        {
            var inputs = new List<double>();
            var weights = new List<double>();

            var emptyLastWeights = new List<double>();

            foreach (var neuron in _hiddenLayerData)
            {
                inputs.Add(neuron.Output);
                weights.Add(GetInitializedWeight());

                emptyLastWeights.Add(0d);
            }

            _outputNeuron = new NeuronModel(inputs, weights)
            {
                LastWeights = emptyLastWeights
            };
        }

        /// <summary>
        /// Обновить веса нейрона на слое.
        /// </summary>
        /// <param name="updatedWeights">Обновлённые веса.</param>
        public void UpdateWeightsOfNeuronInLayer(List<double> updatedWeights) => _outputNeuron.UpdateWeights(updatedWeights);

        /// <summary>
        /// Обновить дельту нейрона на слое.
        /// </summary>
        /// <param name="delta">Дельта</param>
        public void UpdateDeltaOfNeuronInLayer(double delta) => _outputNeuron.Delta = delta;

        /// <summary>
        /// Обновить входные значения у нейрона на слое.
        /// </summary>
        /// <param name="inputs"></param>
        public void UpdateNeuronInputs(List<double> inputs) => _outputNeuron.Inputs = inputs;

        /// <summary>
        /// Инициализация случайных весов.
        /// </summary>
        /// <returns>Возвращает случайный вес.</returns>
        private double GetInitializedWeight() => new Random().NextDouble();

        /// <summary>
        /// Получить значение выходного нейрона.
        /// </summary>
        /// <returns>Возвращает выходной нейрон.</returns>
        public NeuronModel GetOutputNeuron() => _outputNeuron;
    }
}
