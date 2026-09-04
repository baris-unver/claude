export interface PushMessage {
  to: string;
  title: string;
  body: string;
  data?: Record<string, unknown>;
  priority?: 'default' | 'high';
}

export interface PushSender {
  send(messages: PushMessage[]): Promise<void>;
}

/** Expo Push servisi üzerinden bildirim gönderir. Üretimde doğrudan FCM/APNs adaptörü de yazılabilir. */
export class ExpoPushSender implements PushSender {
  constructor(private readonly url: string, private readonly log: (msg: string) => void = () => {}) {}

  async send(messages: PushMessage[]): Promise<void> {
    const valid = messages.filter((m) => m.to.startsWith('ExponentPushToken'));
    if (valid.length === 0) return;
    try {
      const res = await fetch(this.url, {
        method: 'POST',
        headers: { 'content-type': 'application/json', accept: 'application/json' },
        body: JSON.stringify(valid.map((m) => ({ ...m, sound: 'default', priority: m.priority ?? 'high' }))),
      });
      if (!res.ok) this.log(`Push gönderimi başarısız: HTTP ${res.status}`);
    } catch (err) {
      this.log(`Push gönderimi hatası: ${(err as Error).message}`);
    }
  }
}

/** Testler ve bellek içi demo için: gönderilenleri kaydeder. */
export class RecordingPushSender implements PushSender {
  readonly sent: PushMessage[] = [];
  async send(messages: PushMessage[]): Promise<void> {
    this.sent.push(...messages);
  }
}
