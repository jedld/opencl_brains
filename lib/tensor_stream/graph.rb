module TensorStream
  class Graph
    attr_accessor :nodes
    
    def initialize
      @nodes = {}
    end

    def self.get_default_graph
      Thread.current[:tensor_stream_current_graph] || create_default
    end

    def self.create_default
      Thread.current[:tensor_stream_current_graph] = TensorStream::Graph.new
    end

    def add_node(node)
      if @nodes[node.name]
        node.name = uniqunify(node.name)
      end
      @nodes[node.name] = node
    end

    protected

    def uniqunify(name)
      counter = 0
      new_name = name
      begin
        counter +=1
        new_name = "#{name}_#{counter}"
      end while @nodes[new_name]
      new_name
    end
  end
end