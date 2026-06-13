import { LitElement, html } from "lit";
import { EnvDefinition, CommandState } from "../envs/types";

// lucide アイコンを Web Component 内で自動バインドするための処理
// グローバルに用意されている lucide からインポートします
import { createIcons, icons } from "lucide";

export class ManualCommandControls extends LitElement {
  static properties = {
    env: { type: Object },
    command: { type: Object },
    disabled: { type: Boolean },
  };

  declare env: EnvDefinition | null;
  declare command: CommandState | null;
  declare disabled: boolean;

  // Shadow DOM を無効化し、グローバル CSS (styles.css) のテーマ設定をそのまま適用します
  protected createRenderRoot() {
    return this;
  }

  protected updated() {
    // 描画が更新されるたびに Lucide アイコンを再構築
    createIcons({
      icons,
    });
  }

  private handleSliderInput(key: keyof CommandState, event: Event) {
    if (this.disabled) return;
    const input = event.target as HTMLInputElement;
    const value = Number(input.value);

    // 親コンポーネントにカスタムイベントでスライダーの変更を通知
    this.dispatchEvent(
      new CustomEvent("command-change", {
        detail: { key, value },
        bubbles: true,
        composed: true,
      })
    );
  }

  private handleButtonPress(direction: "up" | "down" | "left" | "right", active: boolean) {
    if (this.disabled) return;

    this.dispatchEvent(
      new CustomEvent("direction-press", {
        detail: { direction, active },
        bubbles: true,
        composed: true,
      })
    );
  }

  render() {
    const env = this.env;
    if (!env || !env.policy) {
      return html``;
    }
    return html`
      <style>
        .dpad-container {
          display: flex;
          flex-direction: column;
          align-items: center;
          margin: 16px 0;
          padding: 12px;
          background: #fffdf8;
          border: 1px solid #d6cec0;
          border-radius: 12px;
          box-shadow: inset 0 1px 3px rgba(0,0,0,0.03);
        }

        .dpad-label {
          font-size: 11px;
          font-weight: 750;
          color: #7b6040;
          text-transform: uppercase;
          margin-bottom: 12px;
          letter-spacing: 0.5px;
        }

        .dpad {
          position: relative;
          width: 150px;
          height: 150px;
          background: radial-gradient(circle, #fffdf8 30%, #f4f1eb 100%);
          border: 1.5px solid #c8c0b2;
          border-radius: 50%;
          box-shadow: inset 0 2px 5px rgba(0,0,0,0.04), 0 8px 20px rgba(0,0,0,0.07);
          display: grid;
          grid-template-areas:
            ". up ."
            "left . right"
            ". down .";
          grid-template-columns: 1fr 1fr 1fr;
          grid-template-rows: 1fr 1fr 1fr;
          overflow: hidden;
          touch-action: none;
        }

        .dpad-btn {
          border: none;
          background: transparent;
          display: flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          color: #555a5b;
          transition: all 120ms ease;
          outline: none;
          user-select: none;
          -webkit-user-select: none;
        }

        .dpad-btn svg {
          width: 22px;
          height: 22px;
          stroke-width: 2.2;
          transition: transform 120ms ease;
        }

        .dpad-btn:hover:not(:disabled) {
          background: #dff4ee;
          color: #0c6258;
        }

        .dpad-btn:active:not(:disabled) {
          background: #c9eee6;
        }

        .dpad-btn:disabled {
          cursor: not-allowed;
          opacity: 0.45;
        }

        .dpad-btn[data-dir="up"] { 
          grid-area: up; 
          border-bottom: 0.5px solid rgba(200, 192, 178, 0.4);
        }
        .dpad-btn[data-dir="down"] { 
          grid-area: down; 
          border-top: 0.5px solid rgba(200, 192, 178, 0.4);
        }
        .dpad-btn[data-dir="left"] { 
          grid-area: left; 
          border-right: 0.5px solid rgba(200, 192, 178, 0.4);
        }
        .dpad-btn[data-dir="right"] { 
          grid-area: right; 
          border-left: 0.5px solid rgba(200, 192, 178, 0.4);
        }
      </style>

      <div class="section-title">
        <i data-lucide="gauge"></i>
        <span>Command (Robot)</span>
      </div>

      <div class="dpad-container">
        <div class="dpad-label">D-Pad Controller</div>
        <div class="dpad">
          <button
            @mousedown=${() => this.handleButtonPress("up", true)}
            @mouseup=${() => this.handleButtonPress("up", false)}
            @mouseleave=${() => this.handleButtonPress("up", false)}
            @touchstart=${(e: TouchEvent) => { e.preventDefault(); this.handleButtonPress("up", true); }}
            @touchend=${() => this.handleButtonPress("up", false)}
            ?disabled=${this.disabled}
            class="dpad-btn"
            data-dir="up"
            title="Move Forward"
          >
            <i data-lucide="chevron-up"></i>
          </button>
          
          <button
            @mousedown=${() => this.handleButtonPress("down", true)}
            @mouseup=${() => this.handleButtonPress("down", false)}
            @mouseleave=${() => this.handleButtonPress("down", false)}
            @touchstart=${(e: TouchEvent) => { e.preventDefault(); this.handleButtonPress("down", true); }}
            @touchend=${() => this.handleButtonPress("down", false)}
            ?disabled=${this.disabled}
            class="dpad-btn"
            data-dir="down"
            title="Move Backward"
          >
            <i data-lucide="chevron-down"></i>
          </button>

          <button
            @mousedown=${() => this.handleButtonPress("left", true)}
            @mouseup=${() => this.handleButtonPress("left", false)}
            @mouseleave=${() => this.handleButtonPress("left", false)}
            @touchstart=${(e: TouchEvent) => { e.preventDefault(); this.handleButtonPress("left", true); }}
            @touchend=${() => this.handleButtonPress("left", false)}
            ?disabled=${this.disabled}
            class="dpad-btn"
            data-dir="left"
            title="Turn Left"
          >
            <i data-lucide="chevron-left"></i>
          </button>

          <button
            @mousedown=${() => this.handleButtonPress("right", true)}
            @mouseup=${() => this.handleButtonPress("right", false)}
            @mouseleave=${() => this.handleButtonPress("right", false)}
            @touchstart=${(e: TouchEvent) => { e.preventDefault(); this.handleButtonPress("right", true); }}
            @touchend=${() => this.handleButtonPress("right", false)}
            ?disabled=${this.disabled}
            class="dpad-btn"
            data-dir="right"
            title="Turn Right"
          >
            <i data-lucide="chevron-right"></i>
          </button>
        </div>
      </div>

      ${this.renderSlider("linVelX", "Forward", "m/s")}
      ${this.renderSlider("linVelY", "Lateral", "m/s")}
      ${this.renderSlider("angVelZ", "Yaw", "rad/s")}
    `;
  }

  private renderSlider(key: keyof CommandState, label: string, unit: string) {
    const env = this.env;
    if (!env || !env.policy) {
      return html``;
    }

    const limits = env.policy.commandLimits[key];
    if (!limits) {
      return html``;
    }

    const [min, max] = limits;
    const value = this.command ? this.command[key] ?? 0 : 0;

    return html`
      <label class="range-row" for="${key}-slider">
        <span>${label}</span>
        <output id="${key}-output">${value.toFixed(2)} ${unit}</output>
      </label>
      <input
        id="${key}-slider"
        type="range"
        min=${String(min)}
        max=${String(max)}
        step="0.01"
        .value=${String(value)}
        ?disabled=${this.disabled}
        @input=${(e: Event) => this.handleSliderInput(key, e)}
      />
    `;
  }
}

customElements.define("manual-command-controls", ManualCommandControls);

declare global {
  interface HTMLElementTagNameMap {
    "manual-command-controls": ManualCommandControls;
  }
}
