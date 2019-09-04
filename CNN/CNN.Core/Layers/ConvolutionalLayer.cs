namespace CNN.Core.Layers
{
    using System.Collections.Generic;

    /// <summary>
    /// Класс свёрточного слоя.
    /// </summary>
    public class ConvolutionalLayer
    {
        /// <summary>
        /// Данные входной слоя.
        /// </summary>
        private Dictionary<string, double> _inputLayerData;

        /// <summary>
        /// Данные свёрточного слоя.
        /// </summary>
        private List<Neuron> _convolutionalLayerData;

        /// <summary>
        /// Класс свёрточного слоя.
        /// </summary>
        /// <param name="inputLayerData">Данные входного слоя.</param>
        public ConvolutionalLayer(Dictionary<string, double> inputLayerData)
        {
            _inputLayerData = inputLayerData;
        }

        public void LayerInitialize()
        {
            _convolutionalLayerData = new List<Neuron>();

            foreach (var output in _inputLayerData)
            {

            }
        }
    }
}
