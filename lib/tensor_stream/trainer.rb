module TensorStream
  module Train
    class GradientDescentOptimizer < TensorStream::Operation
      def initialize(learning_rate, options = {})
        @items = [learning_rate]
        @operation = :gradient_descent
        @name = set_name
        @graph = options[:graph] || TensorStream.get_default_graph
        @graph.add_node(self)
      end

      def minimize(cost)
        @items << cost
        self
      end
    end
  end
end