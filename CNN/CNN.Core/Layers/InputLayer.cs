namespace CNN.Core.Layers
{
    using System.Collections.Generic;

    using CNN.BL.Constants;
    using CNN.BL.Enums;

    /// <summary>
    /// Класс входного слоя.
    /// </summary>
    public class InputLayer : Layer
    {
        /// <summary>
        /// Список нейронов входного слоя.
        /// </summary>
        private Dictionary<string, double> _neurons;

        /// <summary>
        /// Полученные данные.
        /// </summary>
        private double[,] _dataSet;

        /// <summary>
        /// Тип слоя.
        /// </summary>
        public override LayerType LayerType => LayerType.Input;

        /// <summary>
        /// Класс входного слоя.
        /// </summary>
        public InputLayer(double[,] dataSet)
        { 
            _dataSet = dataSet;
        }

        /// <summary>
        /// Заполняем входной слой переданными значениями.
        /// </summary>
        public void Initialize()
        {
            _neurons = new Dictionary<string, double>();

            for (var xIndex = 0; xIndex < MatrixConstants.MATRIX_SIZE; ++xIndex)
                for (var yIndex = 0; yIndex < MatrixConstants.MATRIX_SIZE; ++yIndex)
                {
                    var keyString = $"{MatrixConstants.POSITION_IN_X_AXIS}{xIndex}" +
                        $"{MatrixConstants.KEY_SEPARATOR}" +
                        $"{MatrixConstants.POSITION_IN_Y_AXIS}{yIndex}";

                    _neurons.Add(keyString, _dataSet[xIndex, yIndex]);
                }
        }

        /// <summary>
        /// Возвращает список выходов всех нейронов.
        /// </summary>
        /// <returns>Возвращает словарь значений с их позициями.</returns>
        public Dictionary<string, double> GetLayerNeurons() => _neurons;
    }
}
