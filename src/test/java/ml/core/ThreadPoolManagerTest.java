package ml.core;

import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;

class ThreadPoolManagerTest {

    @Test
    void testComputePoolNotNull() {
        assertNotNull(ThreadPoolManager.compute());
        assertFalse(ThreadPoolManager.compute().isShutdown());
    }

    @Test
    void testIoPoolNotNull() {
        assertNotNull(ThreadPoolManager.io());
        assertFalse(ThreadPoolManager.io().isShutdown());
    }

    @Test
    void testSchedulerNotNull() {
        assertNotNull(ThreadPoolManager.scheduler());
        assertFalse(ThreadPoolManager.scheduler().isShutdown());
    }

    @Test
    void testParallelFor() {
        int[] arr = new int[1000];
        ThreadPoolManager.parallelFor(0, 1000, i -> arr[i] = i * 2);
        for (int i = 0; i < 1000; i++) assertEquals(i * 2, arr[i]);
    }

    @Test
    void testParallelism() {
        assertTrue(ThreadPoolManager.parallelism() > 0);
    }

    @AfterAll
    static void cleanup() {
        ThreadPoolManager.shutdownAll();
    }
}
