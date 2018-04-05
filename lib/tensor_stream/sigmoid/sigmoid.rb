
module TensorStream
  class Sigmoid
    attr_accessor :cl_matrix_a, :cols, :rows, :width

    def initialize(context, prog, queue, events = [])
        @context = context
        @queue = queue
        @prog = prog
        @events = events || []
    end

    def program!
      @@matrix_source ||= File.read(File.join(File.dirname(__FILE__), 'opencl', 'sigmoid.cl'))
      @prog = @context.create_program_with_source( @@matrix_source )
      @prog.build
    end

    def execute
      @matrix_c = NArray.sfloat(@cols * @rows)
      @b_matrix_c = @context.create_buffer(@matrix_c.size * @matrix_c.element_size)

      m = OpenCL::Int1::new(@rows)
      n = OpenCL::Int1::new(@cols)
      @events << @prog.sigmoid(@queue, [@rows, @cols], m, n, @cl_matrix_a, @b_matrix_c)
      self
    end

    def result
      { matrix: @b_matrix_c, row: @rows, cols: @cols }
    end

    # generate native buffers from params
    def nativize(matrix_a:)
        native_matrix_a = NArray.sfloat(matrix_a.size * matrix_a[0].size)

        @cols = matrix_a[0].size
        @rows = matrix_a.size

        matrix_a.flatten.each_with_index do |element, index|
            native_matrix_a[index] = element
        end
  
        @cl_matrix_a = @context.create_buffer(native_matrix_a.size * native_matrix_a.element_size, :flags => OpenCL::Mem::COPY_HOST_PTR, :host_ptr => native_matrix_a)
    end

    def rubynize
        @queue.enqueue_read_buffer(@b_matrix_c, @matrix_c, :event_wait_list => @events)
        @queue.finish
        @matrix_c.to_a.each_slice(@cols).collect { |slice| slice }
    end
  end
end