namespace CNN.Core
{
    using System;
    using System.Collections.Generic;

    /// <summary>
    /// Класс нейрона.
    /// </summary>
    public class Neuron
    {
        /// <summary>
        /// Класс нейрона.
        /// </summary>
        /// <param name="inputs">Входные данные.</param>
        /// <param name="weights">Веса.</param>
        public Neuron(List<double> inputs, List<double> weights)
        {
            _inputs = inputs;
            _weights = weights;
        }

        #region Работа с данными

        /// <summary>
        /// Входные данные.
        /// </summary>
        private List<double> _inputs;

        /// <summary>
        /// Веса.
        /// </summary>
        private List<double> _weights;

        /// <summary>
        /// Входные данные.
        /// </summary>
        public List<double> Inputs
        {
            get => _inputs;
            set => _inputs = value;
        }

        /// <summary>
        /// Веса.
        /// </summary>
        public List<double> Weights
        {
            get => _weights;
            set => _weights = value;
        }

        #endregion

        /// <summary>
        /// Выходное значение.
        /// </summary>
        public double Output { get => ActivationFunction(_inputs, _weights); }

        /// <summary>
        /// Функция активации (Сигмоид).
        /// </summary>
        /// <param name="inputs">Входные данные.</param>
        /// <param name="weights">Веса.</param>
        /// <returns>Возвращает нормализованное выходное значение.</returns>
        private double ActivationFunction(List<double> inputs, List<double> weights)
        {
            var summary = 0d;

            for (int index = 0; index < inputs.Count; ++index)
                summary += inputs[index] * weights[index];

            return Math.Pow(1 + Math.Exp(-summary), -1);
        }
    }
}
