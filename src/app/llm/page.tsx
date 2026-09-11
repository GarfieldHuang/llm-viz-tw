import React from 'react';
import { LayerView } from '@/src/llm/LayerView';
import { InfoButton } from '@/src/llm/WelcomePopup';

export const metadata = {
  title: 'LLM 視覺化',
  description: 'LLM 的 3D 動畫視覺化逐步導覽',
};

import { Header } from '@/src/homepage/Header';

export default function Page() {
    return <>
        <Header title="大語言模型 (LLM) 視覺化">
            <InfoButton />
        </Header>
        <LayerView />
        <div id="portal-container"></div>
    </>;
}
