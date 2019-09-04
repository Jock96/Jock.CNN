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
        /// Инициализация ядра фильтра.
        /// </summary>
        public static double[,] Initialize()
        {
            _value = new double[MatrixConstants.FILTER_MATRIX_SIZE,
                MatrixConstants.FILTER_MATRIX_SIZE];

            var random = new Random();

            for (var xIndex = 0; xIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++xIndex)
                for (var yIndex = 0; yIndex < MatrixConstants.FILTER_MATRIX_SIZE; ++yIndex)
                    _value [xIndex, yIndex] = random.NextDouble();

            return _value;
        }

        /// <summary>
        /// Обновляет значение в ядре на указанную матрицу.
        /// </summary>
        /// <param name="matrixOfValues"></param>
        public static void UpdateValues(double[,] matrixOfValues)
        {
            _value = matrixOfValues;
        }
    }
}
