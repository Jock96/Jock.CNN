namespace CNN.Core.Models
{
    using System;

    using CNN.BL.Constants;

    /// <summary>
    /// Класс ядра фильтра.
    /// </summary>
    public static class FilterCoreModel
    {
        /// <summary>
        /// Значение ядра фильтра.
        /// </summary>
        private static double[,] _value;

        /// <summary>
        /// Последнее значение фильтра.
        /// </summary>
        private static double[,] _lastValue;

        /// <summary>
        /// Инициализация ядра фильтра.
        /// </summary>
        public static double[,] Initialize()
        {
            _value = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                MatrixConstants.FILTER_MATRIX_SIZE];

            _lastValue = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                MatrixConstants.FILTER_MATRIX_SIZE];

            var random = new Random();

            for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                {
                    //TODO 0 - 0,01

                    _value[xIndex, yIndex] = random.NextDouble();
                    _lastValue[xIndex, yIndex] = 0d;
                }

            return _value;
        }

        /// <summary>
        /// Последнее значение ядра фильтра.
        /// </summary>
        public static double[,] LastCoreValue => _lastValue;

        /// <summary>
        /// Получить значение ядра фильтра.
        /// </summary>
        public static double[,] GetCore => _value;

        /// <summary>
        /// Обновить ядро фильтра.
        /// </summary>
        /// <param name="newCore">Новое ядро фильтра.</param>
        public static void UpdateCore(double[,] newCore)
        {
            _lastValue = _value;
            _value = newCore;
        }
    }
}
