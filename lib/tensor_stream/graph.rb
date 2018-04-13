module TensorStream
  class Graph
    attr_accessor :nodes, :collections
    
    def initialize
      @nodes = {}
      @collections = {
        :"#{GraphKeys::GLOBAL_VARIABLES}" => []
      }
    end

    def self.get_default_graph
      Thread.current[:tensor_stream_current_graph] || create_default
    end

    def self.create_default
      Thread.current[:tensor_stream_current_graph] = TensorStream::Graph.new
    end

    def get_collection(name, options = {})
      @collections[name.to_sym]
    end

    def add_to_collection(collection_name, val)
      @collections[collection_name.to_sym] ||= []
      @collections[collection_name.to_sym] << val
    end

    def add_node(node)
      if @nodes[node.name]
        node.name = uniqunify(node.name)
      end

      @nodes[node.name] = node
    end

    def add_variable(node, options = {})
      fail "duplicate variable detected #{node.name} and reuse=false in current scope" if @nodes[node.name] && !options[:reuse]

      add_to_collection(GraphKeys::GLOBAL_VARIABLES, node)
      @nodes[node.name] = node
    end

    def control_dependencies(dependencies = [], &block)
      
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