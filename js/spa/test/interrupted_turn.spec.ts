import { test, expect } from "@playwright/test";
import { spawn, type ChildProcess } from "node:child_process";
import { writeFileSync, rmSync } from "node:fs";
import { createServer } from "node:net";
import { join } from "node:path";

// U10 (R10, R11): when the replica running a turn dies, the stream ends in a
// visible, retryable error and the client never resubmits on its own. A user
// resend succeeds. Killing a single-replica app mid-stream reproduces the same
// client-visible failure a scaled-in replica causes.
//
// Spawns the venv python directly rather than the `gradio` CLI, which is not on
// PATH on Windows CI runners.
const ROOT = join(process.cwd(), "..", "..");
const demo_file = join(process.cwd(), "interrupted_turn_demo.py");

const DEMO = `
import time
import gradio as gr

def slow(state):
    for i in range(20):
        time.sleep(0.5)
        yield f"step {i}"

with gr.Blocks() as demo:
    state = gr.State("kept")
    out = gr.Textbox(label="Stream Output")
    run = gr.Button("Run")
    run.click(slow, [state], [out])

demo.queue().launch(server_name="127.0.0.1", server_port=PORT)
`;

async function free_port(): Promise<number> {
	// Ask the OS for an unused port by opening and closing a server.
	return new Promise((resolve) => {
		const server = createServer();
		server.listen(0, "127.0.0.1", () => {
			const { port } = server.address() as { port: number };
			server.close(() => resolve(port));
		});
	});
}

async function launch_demo(): Promise<{ port: number; proc: ChildProcess }> {
	const port = await free_port();
	return launch_demo_on(port);
}

async function launch_demo_on(
	port: number,
): Promise<{ port: number; proc: ChildProcess }> {
	writeFileSync(demo_file, DEMO.replace("PORT", String(port)));
	return new Promise((resolve, reject) => {
		const proc = spawn(
			join(ROOT, ".venv", "Scripts", "python.exe"),
			[demo_file],
			{
				cwd: process.cwd(),
				env: { ...process.env, PYTHONUNBUFFERED: "true", PYTHONPATH: ROOT },
			},
		);
		let out = "";
		const on_data = (data: Buffer) => {
			out += data.toString();
			if (out.includes("Running on local URL:")) resolve({ port, proc });
		};
		proc.stdout!.on("data", on_data);
		proc.stderr!.on("data", on_data);
		proc.on("exit", (code) =>
			reject(new Error(`demo exited early (${code}): ${out}`)),
		);
		setTimeout(() => reject(new Error(`demo did not start: ${out}`)), 30000);
	});
}

test.afterAll(() => {
	try {
		rmSync(demo_file);
	} catch {
		// already gone
	}
});

test("an interrupted turn shows a retryable error and is not auto-resubmitted", async ({
	page,
}) => {
	test.setTimeout(90 * 1000);
	const { port, proc } = await launch_demo();

	const submits: string[] = [];
	await page.route("**/queue/join*", async (route) => {
		submits.push(route.request().url());
		await route.continue();
	});

	try {
		await page.goto(`http://localhost:${port}`);
		// Smoke: the app actually mounted before we exercise behavior.
		await expect(page.getByRole("button", { name: "Run" })).toBeVisible({
			timeout: 15 * 1000,
		});

		await page.getByRole("button", { name: "Run" }).click();
		await expect(page.getByLabel("Stream Output")).toHaveValue(/step \d/, {
			timeout: 15 * 1000,
		});

		// Fail the turn the way a lost replica does: drop the in-flight stream.
		await page.route("**/queue/data*", (route) => route.abort());
		proc.kill();

		await expect(page.getByText(/Connection Lost|Error/i).first()).toBeVisible({
			timeout: 20 * 1000,
		});

		// Nothing resubmitted the turn automatically.
		const before = submits.length;
		await page.waitForTimeout(5000);
		expect(submits.length).toBe(before);
	} finally {
		if (!proc.killed) proc.kill();
	}
});

test("a user resend after reconnect reaches a surviving replica", async ({
	page,
}) => {
	test.setTimeout(90 * 1000);
	const first = await launch_demo();
	const port = first.port;

	let second: { port: number; proc: ChildProcess } | null = null;
	try {
		await page.goto(`http://localhost:${port}`);
		const run = page.getByRole("button", { name: "Run" });
		await expect(run).toBeVisible({ timeout: 15 * 1000 });

		await run.click();
		await expect(page.getByLabel("Stream Output")).toHaveValue(/step \d/, {
			timeout: 15 * 1000,
		});

		// The replica dies mid-turn and the client latches into connection_lost;
		// it will not accept another submission on this page.
		first.proc.kill();
		await expect(page.getByText(/Connection Lost|Error/i).first()).toBeVisible({
			timeout: 20 * 1000,
		});

		// A survivor comes up on the same port and the client reconnects, which
		// reloads the page -- the documented recovery path.
		second = await launch_demo_on(port);
		await expect(async () => {
			expect(page.url()).toBe(`http://localhost:${port}/`);
		}).toPass({ timeout: 20 * 1000 });
		await expect(page.getByRole("button", { name: "Run" })).toBeVisible({
			timeout: 20 * 1000,
		});

		// The user resends on the recovered page and it succeeds end to end.
		await page.getByRole("button", { name: "Run" }).click();
		await expect(page.getByLabel("Stream Output")).toHaveValue(/step \d/, {
			timeout: 20 * 1000,
		});
	} finally {
		if (!first.proc.killed) first.proc.kill();
		if (second && !second.proc.killed) second.proc.kill();
	}
});
